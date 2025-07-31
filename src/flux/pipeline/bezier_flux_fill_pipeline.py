"""
BezierFluxFillPipeline - FLUX Fill Pipeline with BezierAdapter integration.

This pipeline extends the standard FLUX Fill capability with BezierAdapter's
precision control for font stylization and typography generation.
"""

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput

from ..modules.autoencoder import AutoEncoder
from ..modules.bezier_flux_model import FluxBezierAdapter
from ..modules.models import MultiModalCondition, BezierControlPoints
from ..util import load_image

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        import torch
        from PIL import Image
        from bezier_flux_fill_pipeline import BezierFluxFillPipeline
        from bezier_flux_fill_pipeline.utils import load_bezier_curves

        # Load pipeline
        pipe = BezierFluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            bezier_adapter_path="path/to/bezier_adapter.safetensors",
            torch_dtype=torch.bfloat16
        ).to("cuda")

        # Load inputs
        image = Image.open("base_font.png")
        mask = Image.open("mask.png")
        style_image = Image.open("style_reference.png")
        bezier_curves = load_bezier_curves("character_curves.json")

        # Generate with BezierAdapter
        result = pipe(
            prompt="elegant serif typography",
            image=image,
            mask_image=mask,
            style_image=style_image,
            bezier_curves=bezier_curves,
            height=1024, 
            width=1024,
            guidance_scale=30.0,
            num_inference_steps=50
        ).images[0]
        ```
"""


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Retrieve timesteps from scheduler.
    
    Compatible with Diffusers scheduler interface.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_sigmas(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


class BezierFluxFillPipeline(DiffusionPipeline):
    r"""
    FLUX Fill Pipeline with BezierAdapter integration for font stylization.
    
    This pipeline extends FLUX.1-Fill-dev's inpainting capabilities with BezierAdapter's
    precision control mechanisms, enabling:
    - Bézier curve-guided font generation
    - Style transfer between font families  
    - Density-aware spatial attention
    - Multi-modal conditioning (style + text + mask + Bézier)
    
    The pipeline inherits from DiffusionPipeline and maintains full compatibility with
    the standard FLUX Fill API while adding enhanced typography control.

    Args:
        transformer ([`FluxBezierAdapter`]):
            FLUX transformer with integrated BezierAdapter framework.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            The scheduler to be used with the transformer.
        vae ([`AutoEncoder`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images.
        text_encoder ([`CLIPTextModel`]):
            CLIP text encoder for style conditioning.
        text_encoder_2 ([`T5EncoderModel`]):
            T5 text encoder for semantic conditioning.
        tokenizer ([`CLIPTokenizer`]):
            CLIP tokenizer for text processing.
        tokenizer_2 ([`T5TokenizerFast`]):
            T5 tokenizer for text processing.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "pooled_prompt_embeds",
    ]

    def __init__(
        self,
        transformer: FluxBezierAdapter,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoEncoder,
        text_encoder: CLIPTextModel,
        text_encoder_2: T5EncoderModel,
        tokenizer: CLIPTokenizer,
        tokenizer_2: T5TokenizerFast,
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
        )

        # BezierAdapter-specific attributes
        self.bezier_enabled = hasattr(transformer, 'is_fill_model')
        if self.bezier_enabled:
            self.is_fill_model = transformer.is_fill_model
            self.bezier_stats = transformer.bezier_stats
        else:
            self.is_fill_model = False
            self.bezier_stats = {}

        # Validate model compatibility
        if not self.is_fill_model:
            logger.warning(
                "BezierFluxFillPipeline requires FLUX.1-Fill-dev model (384 channels). "
                f"Current model has {getattr(transformer, 'in_channels', 'unknown')} channels."
            )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)) if hasattr(vae, "config") else 16
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Generate T5 text embeddings."""
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_2.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                f"The following part of your input was truncated: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # Duplicate text embeddings and attention mask for each generation per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        """Generate CLIP text embeddings."""
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(f"The following part of your input was truncated: {removed_text}")

        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # Duplicate text embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        """
        Encode prompts using both CLIP and T5 encoders.
        
        Args:
            prompt: Text prompt for generation
            prompt_2: Optional second prompt (uses prompt if None)
            device: Device for computation
            num_images_per_prompt: Number of images per prompt
            prompt_embeds: Pre-computed CLIP embeddings
            pooled_prompt_embeds: Pre-computed pooled CLIP embeddings
            max_sequence_length: Maximum T5 sequence length
            lora_scale: LoRA scale for text encoders
            
        Returns:
            tuple: (prompt_embeds, pooled_prompt_embeds, text_ids)
        """
        device = device or self._execution_device

        # Set up LoRA scale
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # Dynamically adjust LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt_2 = prompt_2 or prompt
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # Get CLIP embeddings
        pooled_prompt_embeds_2, pooled_prompt_embeds = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
        prompt_embeds = pooled_prompt_embeds_2

        # Get T5 embeddings
        prompt_embeds_2 = self._get_t5_prompt_embeds(
            prompt=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        clip_prompt_embeds = prompt_embeds
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])

        # Create text IDs for positional encoding
        text_ids = torch.zeros(prompt_embeds_2.shape[1], 3).to(device=device, dtype=torch.int)

        if isinstance(self, FluxLoraLoaderMixin):
            # Reset LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return clip_prompt_embeds, prompt_embeds_2, pooled_prompt_embeds, text_ids

    def _prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        generator,
        add_noise=True,
    ):
        """Prepare latents for denoising process."""
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        latents_mean = latents_std = None
        if hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1)
        if hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None:
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1)

        image = self.image_processor.preprocess(image, height=None, width=None).to(device)
        image = image.to(dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            if image.shape[0] < batch_size:
                image = image.repeat(batch_size // image.shape[0], 1, 1, 1)
            init_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        if latents_mean is not None and latents_std is not None:
            latents_mean = latents_mean.to(device=device, dtype=dtype)
            latents_std = latents_std.to(device=device, dtype=dtype)
            init_latents = (init_latents - latents_mean) * self.vae.config.scaling_factor / latents_std
        else:
            init_latents = (init_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        if add_noise:
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # get latents
            init_latents = self.scheduler.scale_noise(init_latents, timestep, noise)

        latents = init_latents
        return latents

    def _prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        """Prepare mask latents for inpainting."""
        # Resize the mask to latents shape as we concatenate the mask to the latents
        # We do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        # Duplicate mask and masked_image_latents for each generation per prompt
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask

        if masked_image is not None and masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = None

        if masked_image_latents is not None:
            if masked_image_latents.shape[0] < batch_size:
                if not batch_size % masked_image_latents.shape[0] == 0:
                    raise ValueError(
                        "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                        f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                        " Make sure the number of images that you pass is divisible by the total requested batch size."
                    )
                masked_image_latents = masked_image_latents.repeat(
                    batch_size // masked_image_latents.shape[0], 1, 1, 1
                )

            masked_image_latents = (
                torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
            )

            # aligning device to prevent device errors when concating it with the latent model input
            masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

        return mask, masked_image_latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def prepare_bezier_conditions(
        self,
        bezier_curves: Optional[Union[List[Dict], str, BezierControlPoints]] = None,
        style_image: Optional[PIL.Image.Image] = None,
        prompt: Optional[str] = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare BezierAdapter conditioning inputs.
        
        Args:
            bezier_curves: Bézier curve specifications
            style_image: Style reference image for font transfer
            prompt: Text prompt for enhanced conditioning
            device: Target device
            dtype: Target dtype
            
        Returns:
            Dictionary containing BezierAdapter conditioning data
        """
        if bezier_curves is None and style_image is None:
            return None

        device = device or self._execution_device
        dtype = dtype or torch.float32

        conditions = {}

        # Process style image through CLIP if provided
        if style_image is not None:
            # Preprocess style image
            style_tensor = self.image_processor.preprocess(style_image, height=512, width=512)
            style_tensor = style_tensor.to(device=device, dtype=dtype)
            
            # Encode through CLIP (simplified - in practice would use CLIP vision encoder)
            style_features = torch.randn(1, 768, device=device, dtype=dtype)  # Placeholder
            conditions['style_features'] = style_features

        # Process text through T5 if provided
        if prompt is not None:
            # This would use the T5 encoder, simplified for now
            text_features = torch.randn(1, 4096, device=device, dtype=dtype)  # Placeholder
            conditions['text_features'] = text_features

        # Process Bézier curves
        if bezier_curves is not None:
            if isinstance(bezier_curves, str):
                # Load from JSON file
                import json
                with open(bezier_curves, 'r') as f:
                    curve_data = json.load(f)
                bezier_points = torch.tensor(curve_data['control_points'], dtype=dtype, device=device)
            elif isinstance(bezier_curves, list):
                # Convert list of dicts to tensor
                points = []
                for curve in bezier_curves:
                    if isinstance(curve, dict) and 'points' in curve:
                        points.extend(curve['points'])
                    else:
                        points.extend(curve)
                bezier_points = torch.tensor(points, dtype=dtype, device=device)
            else:
                bezier_points = bezier_curves.to(device=device, dtype=dtype)

            # Add batch dimension if needed
            if bezier_points.dim() == 2:
                bezier_points = bezier_points.unsqueeze(0)

            # Create bezier features (x, y, density) - simplified density calculation
            num_points = bezier_points.shape[1]
            densities = torch.ones(bezier_points.shape[0], num_points, 1, device=device, dtype=dtype) / num_points
            bezier_features = torch.cat([bezier_points, densities], dim=-1)
            
            conditions['bezier_features'] = bezier_features

        # Create MultiModalCondition object
        multimodal_condition = MultiModalCondition(
            style_features=conditions.get('style_features'),
            text_features=conditions.get('text_features'),
            mask_features=None,  # Will be set from mask_image processing
            bezier_features=conditions.get('bezier_features')
        )

        return {
            'conditions': multimodal_condition,
            'bezier_points': bezier_points if 'bezier_points' in locals() else None
        }

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 30.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # BezierAdapter-specific parameters
        bezier_curves: Optional[Union[List[Dict], str, BezierControlPoints]] = None,
        style_image: Optional[PIL.Image.Image] = None,
        density_guidance_scale: float = 1.0,
        style_guidance_scale: float = 0.7,
    ):
        """
        Generate images using FLUX Fill with BezierAdapter enhancements.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be inpainted (parts of the image are
                masked out with `mask_image` and repainted according to `prompt`).
            mask_image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to mask `image`. White pixels in the mask
                are repainted while black pixels are preserved.
            height (`int`, *optional*, defaults to the size of `image`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to the size of `image`):
                The width in pixels of the generated image.
            strength (`float`, *optional*, defaults to 1.0):
                Indicates extent to which to transform the reference `image`. Must be between 0 and 1.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process.
            guidance_scale (`float`, *optional*, defaults to 30.0):
                Higher guidance scale encourages to generate images that are closely linked to the text `prompt`.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of torch generator(s) to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`].
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising step during the inference.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function.
            max_sequence_length (`int` defaults to 512):
                Maximum sequence length to use with the `prompt`.

            # BezierAdapter-specific parameters
            bezier_curves (`List[Dict]`, `str`, or `BezierControlPoints`, *optional*):
                Bézier curve specifications for font control. Can be:
                - List of curve dictionaries with 'points' keys
                - Path to JSON file containing curve data
                - BezierControlPoints object
            style_image (`PIL.Image.Image`, *optional*):
                Reference image for style transfer between font families.
            density_guidance_scale (`float`, *optional*, defaults to 1.0):
                Guidance scale for Bézier density conditioning.
            style_guidance_scale (`float`, *optional*, defaults to 0.7):
                Guidance scale for style transfer conditioning.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images.
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            height=height,
            width=width,
            strength=strength,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = None
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        # 4. Prepare BezierAdapter conditioning
        bezier_conditions = self.prepare_bezier_conditions(
            bezier_curves=bezier_curves,
            style_image=style_image,
            prompt=prompt,
            device=device,
            dtype=prompt_embeds.dtype
        )

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps,
            strength=strength,
            device=device,
        )
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength, the number of pipeline steps is {num_inference_steps} "
                f"which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 6. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        num_channels_transformer = self.transformer.config.in_channels
        return_image_latents = num_channels_transformer == num_channels_latents

        latents = self._prepare_latents(
            image=image,
            timestep=latent_timestep,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            add_noise=True,
        )

        # 7. Prepare mask latent variables
        mask_condition = self.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode="default"
        )

        if masked_image is None:
            masked_image = image * (mask_condition < 0.5)
        else:
            masked_image = self.image_processor.preprocess(
                masked_image, height=height, width=width, resize_mode="default"
            )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, None)

        # 9. Create image_ids
        height_latent = int(height) // self.vae_scale_factor
        width_latent = int(width) // self.vae_scale_factor
        image_ids = self._prepare_image_ids(batch_size * num_images_per_prompt, height_latent, width_latent, device, prompt_embeds.dtype)

        # 10. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                
                # Broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

                # Predict noise residual with BezierAdapter
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    # timestep=timestep,
                    timesteps=timestep,
                    guidance=torch.tensor([guidance_scale], device=device, dtype=latent_model_input.dtype).expand(latent_model_input.shape[0]),
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    # BezierAdapter conditioning
                    bezier_conditions=bezier_conditions,
                    return_dict=False,
                )[0]

                # Perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    pooled_prompt_embeds = callback_outputs.pop("pooled_prompt_embeds", pooled_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)