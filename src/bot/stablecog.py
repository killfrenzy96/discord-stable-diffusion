import traceback
from asyncio import AbstractEventLoop
from threading import Thread

import ntpath
import requests
import asyncio
import discord
from discord.ext import commands
from typing import Optional
from io import BytesIO
from PIL import Image
from discord import option
import random
import time
import torch

from src.stablediffusion.text2image_compvis import Text2Image
import src.bot.shanghai

embed_color = discord.Colour.from_rgb(215, 195, 134)
# checkpoint_names = []


class DreamQueueObject:
    def __init__(self, ctx, checkpoint, prompt, negative, height, width, guidance_scale, steps, seed, strength,
                 init_image, mask_image, sampler_name, command_str, is_batch):
        self.ctx = ctx
        self.checkpoint = checkpoint
        self.prompt = prompt
        self.negative = negative
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.seed = seed
        self.strength = strength
        self.init_image = init_image
        self.mask_image = mask_image
        self.sampler_name = sampler_name
        self.command_str = command_str
        self.is_batch = is_batch


class UploadQueueObject:
    def __init__(self, ctx, content, embed, file):
        self.ctx = ctx
        self.content = content
        self.embed = embed
        self.file = file


class Text2ImageCheckpoint:
    def __init__(self, path: str):
        self.path = path
        head, tail = ntpath.split(path)
        self.name = tail.split('.')[0]


class Text2ImageModel:
    def __init__(self, checkpoint: Text2ImageCheckpoint):
        self.checkpoint = checkpoint
        self.text2image_model = Text2Image(model_path=checkpoint.path)
        self.last_used = time.time()



class StableCog(commands.Cog, name='Stable Diffusion', description='Create images from natural language.'):
    def __init__(self, bot):
        self.dream_thread = Thread()
        self.upload_thread = Thread()

        self.checkpoints = []
        checkpoint_paths = bot.args.model_path.split('|')

        for path in checkpoint_paths:
            self.checkpoints.append(Text2ImageCheckpoint(path))
            print("Checkpoint: " + path)

        self.checkpoint_main = self.checkpoints[0]
        print("Default main checkpoint: " + self.checkpoint_main.name)

        self.checkpoint_anime = self.checkpoint_main
        for checkpoint in self.checkpoints:
            if checkpoint.name == 'waifu_diffusion':
                self.checkpoint_anime = checkpoint
        print("Default anime checkpoint: " + self.checkpoint_anime.name)

        # for checkpoint in self.checkpoints:
        #     checkpoint_names.append(checkpoint.name)

        models_loaded_length = int(bot.args.model_cache)
        if models_loaded_length == 0: models_loaded_length = 1
        print("Maximum models to keep in VRAM: " + str(models_loaded_length))

        self.models_loaded = [None] * models_loaded_length
        # self.models_loaded[0] = Text2ImageModel(self.checkpoint_main)

        self.dream_event_loop = asyncio.get_event_loop()
        self.upload_event_loop = asyncio.get_event_loop()
        self.dream_queue = []
        self.upload_queue = []
        self.bot = bot

    def load_model(self, checkpoint_name: str):
        # Use checkpoint is already VRAM
        for model in self.models_loaded:
            if model == None:
                continue
            if model.checkpoint.name == checkpoint_name:
                model.last_used = time.time()
                return model.text2image_model

        # Load checkpoint into VRAM
        for checkpoint in self.checkpoints:
            if checkpoint.name == checkpoint_name:
                # Unload oldest model
                model_index = 0
                for index, model in enumerate(self.models_loaded):
                    if model == None:
                        model_index = index
                        break
                    if model.last_used < self.models_loaded[model_index].last_used:
                        model_index = index

                if self.models_loaded[model_index] != None:
                    print("Unloading model [" + str(model_index) + "]: " + self.models_loaded[model_index].checkpoint.name)
                    self.models_loaded[model_index] = None
                    torch.cuda.empty_cache()

                # Load model
                print("Loading model [" + str(model_index) + "]: " + checkpoint.name)
                self.models_loaded[model_index] = Text2ImageModel(checkpoint)
                return self.models_loaded[model_index].text2image_model

    @commands.slash_command(name='dream', description='Create an image.')
    @option(
        'prompt',
        str,
        description='A prompt to condition the model with.',
        required=True,
    )
    @option(
        'negative',
        str,
        description='A negative prompt to uncondition the model with.',
        required=False,
        default=''
    )
    @option(
        'checkpoint',
        str,
        description='Which checkpoint model to use for generation',
        required=False,
        choices=[ # checkpoint_names,
            'stable_diffusion',
            'stable_diffusion_inpainting',
            'waifu_diffusion',
            'waifu_diffusion_merge_hitten',
            'waifu_diffusion_merge_stable_diffusion',
            'waifu_diffusion_merge_trinart_characters',
            'trinart',
            'trinart_characters',
            'hitten_girl_anime'
        ],
        default=None
    )
    @option(
        'height',
        int,
        description='Height of the generated image.',
        required=False,
        choices=[x for x in range(128, 769, 64)]
    )
    @option(
        'width',
        int,
        description='Width of the generated image.',
        required=False,
        choices=[x for x in range(128, 769, 64)]
    )
    @option(
        'guidance_scale',
        float,
        description='Classifier-Free Guidance scale',
        required=False,
    )
    @option(
        'steps',
        int,
        description='The amount of steps to sample the model',
        required=False,
        choices=[x for x in range(5, 55, 5)]
    )
    @option(
        'sampler',
        str,
        description='The sampler to use for generation',
        required=False,
        choices=['ddim', 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms', 'plms'],
        default=None
    )
    @option(
        'seed',
        int,
        description='The seed to use for reproduceability',
        required=False,
    )
    @option(
        'strength',
        float,
        description='The strength (0.0 to 1.0) used to apply the prompt to the init_image/mask_image'
    )
    @option(
        'init_image',
        discord.Attachment,
        description='The image to initialize the latents with for denoising',
        required=False,
    )
    @option(
        'mask_image',
        discord.Attachment,
        description='The mask image to use for inpainting',
        required=False,
    )
    @option(
        'batch',
        int,
        description='The amount of images to generate',
        required=False,
        choices=[1,2,3,4,5,6,7,8],
        default=1
    )
    async def dream_handler(self, ctx: discord.ApplicationContext, *, prompt: str,
                            negative: Optional[str] = '',
                            checkpoint: Optional[str] = None,
                            height: Optional[int] = 512,
                            width: Optional[int] = 512,
                            guidance_scale: Optional[float] = 7.0,
                            steps: Optional[int] = 20,
                            sampler: Optional[str] = None,
                            seed: Optional[int] = -1,
                            strength: Optional[float] = None,
                            init_image: Optional[discord.Attachment] = None,
                            mask_image: Optional[discord.Attachment] = None,
                            batch: Optional[int] = 1):
        print(f'Request -- {ctx.author.name}#{ctx.author.discriminator}')

        if seed == -1:
            seed = random.randint(0, 0xFFFFFFFF)

        # clean prompt input
        prompt.replace(':', ' ')
        negative.replace(':', ' ')
        prompt.replace('`', ' ')
        negative.replace('`', ' ')

        # Make sure checkpoint is a valid value
        checkpointFound = False
        for cp in self.checkpoints:
            if cp.name == checkpoint:
                checkpointFound = True
        if checkpointFound == False:
            checkpoint = None

        # Checkpoint autoselect
        if checkpoint == None:
            checkpoint = self.checkpoints[0].name
            if 'anime' in prompt or 'waifu' in prompt:
                checkpoint = self.checkpoint_anime.name

        # Set sampler
        if sampler == None:
            sampler = 'ddim'
            if 'stable' in checkpoint:
                sampler = 'ddim'
            elif 'waifu' in checkpoint:
                sampler = 'k_euler_a'
            elif 'hitten' in checkpoint or 'trinart' in checkpoint:
                sampler = 'k_euler'

        # Setup command string
        def get_command_str():
            command_str = '/dream'
            command_str = command_str + f' prompt:{prompt} negative:{negative} checkpoint:{checkpoint} height:{str(height)} width:{width} guidance_scale:{guidance_scale} steps:{steps} sampler:{sampler} seed:{seed} batch:{batch}'

            # Set images
            if init_image:
                command_str = command_str + f' init_image:{init_image.url}'

            if mask_image:
                command_str = command_str + f' mask_image:{mask_image.url}'

            if init_image or mask_image:
                if strength == None: strength = 0.75
                command_str = command_str + f' strength:{strength}'

            return command_str

        content = ''
        ephemeral = False

        user_already_in_queue = 0
        for queue_object in self.dream_queue:
            if queue_object.ctx.author.id == ctx.author.id:
                user_already_in_queue += 1

        if user_already_in_queue > 2:
            content=f'<@{ctx.author.id}> Please wait for your current images to finish generating before generating a new image'
            ephemeral=True

        else:
            queue_length = len(self.dream_queue)

            command_str = get_command_str()
            print(command_str)

            await self.process_dream(DreamQueueObject(
                ctx, checkpoint, prompt, negative, height, width, guidance_scale, steps, seed,
                strength, init_image, mask_image, sampler, command_str, False
            ))

            batch_count = 1
            while batch_count < batch:
                seed += 1
                batch_count += 1
                command_str = get_command_str()
                self.dream_queue.append(DreamQueueObject(
                    ctx, checkpoint, prompt, negative, height, width, guidance_scale, steps, seed,
                    strength, init_image, mask_image, sampler, command_str, True
                ))

            # content=f'Dreaming for <@{ctx.author.id}> - Queue Position: ``{len(self.queue)}`` - ``{command_str}``'
            content=f'<@{ctx.author.id}> Dreaming - Queue Position: ``{queue_length}``'

            if batch:
                content += f' - Batch: ``{batch}``'

        try:
            await ctx.send_response(content=content, ephemeral=ephemeral)
        except:
            await ctx.channel.send(content)

    async def process_dream(self, dream_queue_object: DreamQueueObject):
        if self.dream_thread.is_alive():
            self.dream_queue.append(dream_queue_object)
        else:
            self.dream_thread = Thread(target=self.dream,
                                    args=(self.dream_event_loop, dream_queue_object))
            self.dream_thread.start()

    def dream(self, dream_event_loop: AbstractEventLoop, queue_object: DreamQueueObject):
        try:
            start_time = time.time()

            model: Text2Image = self.load_model(queue_object.checkpoint)

            if (queue_object.init_image is None) and (queue_object.mask_image is None):
                samples, seed = model.dream(queue_object.prompt, queue_object.negative, queue_object.steps, False, False, 0.0,
                                                            1, 1, queue_object.guidance_scale, queue_object.seed,
                                                            queue_object.height, queue_object.width, False,
                                                            queue_object.sampler_name)
            elif queue_object.init_image is not None:
                image = Image.open(requests.get(queue_object.init_image.url, stream=True).raw).convert('RGB')
                samples, seed = model.translation(queue_object.prompt, queue_object.negative, image, queue_object.steps, 0.0,
                                                                  0,
                                                                  0, queue_object.guidance_scale,
                                                                  queue_object.strength, queue_object.seed,
                                                                  queue_object.height, queue_object.width,
                                                                  queue_object.sampler_name)
            else:
                image = Image.open(requests.get(queue_object.init_image.url, stream=True).raw).convert('RGB')
                mask = Image.open(requests.get(queue_object.mask_image.url, stream=True).raw).convert('RGB')
                samples, seed = model.inpaint(queue_object.prompt, queue_object.negative, image, mask, queue_object.steps, 0.0,
                                                              1, 1, queue_object.guidance_scale,
                                                              denoising_strength=queue_object.strength,
                                                              seed=queue_object.seed, height=queue_object.height,
                                                              width=queue_object.width,
                                                              sampler_name=queue_object.sampler_name)
            end_time = time.time()
            del model

            def upload_dream():
                async def run():
                    with BytesIO() as buffer:
                        samples[0].save(buffer, 'PNG')
                        buffer.seek(0)
                        # embed = discord.Embed()
                        # embed.colour = embed_color
                        # embed.add_field(name='command', value=f'``{queue_object.command_str}``', inline=False)
                        # embed.add_field(name='compute used', value='``{0:.3f}`` seconds'.format(end_time - start_time),
                        #                 inline=False)
                        # embed.add_field(name='react', value='React with ❌ to delete your own generation\nReact with 🔁 to generate this picture again\n' +
                        #     'Compute time: ' + '``{0:.3f}`` seconds\n'.format(end_time - start_time))
                        # fix errors if user doesn't have pfp
                        # if queue_object.ctx.author.avatar is None:
                        #     embed.set_footer(
                        #         text=f'{queue_object.ctx.author.name}#{queue_object.ctx.author.discriminator}')
                        # else:
                        #     embed.set_footer(
                        #         text=f'{queue_object.ctx.author.name}#{queue_object.ctx.author.discriminator}',
                        #         icon_url=queue_object.ctx.author.avatar.url)
                        embed = None

                        content = ''
                        if queue_object.is_batch:
                            content = f'<@{queue_object.ctx.author.id}>'
                        else:
                            content = f'<@{queue_object.ctx.author.id}> ``{queue_object.command_str}``'

                        file = discord.File(fp=buffer, filename=f'{seed}.png')

                        # dream_event_loop.create_task(
                        #     queue_object.ctx.channel.send(content=content, embed=None, file=file)
                        # )

                        await self.process_upload(UploadQueueObject(
                            ctx=queue_object.ctx, content=content, embed=embed, file=file
                        ))
                asyncio.run(run())
            Thread(target=upload_dream, daemon=True).start()

        except Exception as e:
            embed = discord.Embed(title='txt2img failed', description=f'{e}\n{traceback.print_exc()}',
                                  color=embed_color)
            dream_event_loop.create_task(queue_object.ctx.channel.send(embed=embed))
        if self.dream_queue:
            # event_loop.create_task(self.process_dream(self.queue.pop(0)))
            self.dream(dream_event_loop, self.dream_queue.pop(0))

    async def process_upload(self, upload_queue_object: UploadQueueObject):
        if self.upload_thread.is_alive():
            self.upload_queue.append(upload_queue_object)
        else:
            self.upload_thread = Thread(target=self.upload,
                                    args=(self.upload_event_loop, upload_queue_object))
            self.upload_thread.start()

    def upload(self, upload_event_loop: AbstractEventLoop, upload_queue_object: UploadQueueObject):
        upload_event_loop.create_task(
            upload_queue_object.ctx.channel.send(
                content=upload_queue_object.content,
                embed=upload_queue_object.embed,
                file=upload_queue_object.file
            )
        )

        if self.upload_queue:
            # event_loop.create_task(self.process_dream(self.queue.pop(0)))
            self.upload(upload_event_loop, self.upload_queue.pop(0))

def setup(bot):
    src.bot.shanghai._stableCog = StableCog(bot)
    bot.add_cog(src.bot.shanghai._stableCog)
