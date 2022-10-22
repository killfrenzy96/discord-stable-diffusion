import traceback
from asyncio import AbstractEventLoop
from threading import Thread

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

from src.stablediffusion.text2image_compvis import Text2Image
import src.bot.shanghai

embed_color = discord.Colour.from_rgb(215, 195, 134)


class QueueObject:
    def __init__(self, ctx, prompt, negative, height, width, guidance_scale, steps, seed, strength,
                 init_image, mask_image, sampler_name, command_str):
        self.ctx = ctx
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


class StableCog(commands.Cog, name='Stable Diffusion', description='Create images from natural language.'):
    def __init__(self, bot):
        self.dream_thread = Thread()
        self.text2image_model_stable = Text2Image(model_path=bot.args.model_path)
        self.text2image_model_waifu = Text2Image(model_path=bot.args.model_path_waifu)
        self.text2image_model = self.text2image_model_stable
        self.event_loop = asyncio.get_event_loop()
        self.queue = []
        self.bot = bot


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
        choices=['stable_diffusion', 'waifu_diffusion'],
        default=None
    )
    @option(
        'height',
        int,
        description='Height of the generated image.',
        required=False,
        choices=[x for x in range(128, 768, 128)]
    )
    @option(
        'width',
        int,
        description='Width of the generated image.',
        required=False,
        choices=[x for x in range(128, 768, 128)]
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
                            mask_image: Optional[discord.Attachment] = None):
        print(f'Request -- {ctx.author.name}#{ctx.author.discriminator}')

        if seed == -1:
            seed = random.randint(0, 0xFFFFFFFF)

        # clean prompt input
        prompt.replace(':', ' ')
        negative.replace(':', ' ')
        prompt.replace('`', ' ')
        negative.replace('`', ' ')

        if checkpoint == None:
            promptLower = prompt.lower()
            if 'anime' in promptLower or 'waifu' in promptLower:
                checkpoint = 'waifu_diffusion'
            else:
                checkpoint = 'stable_diffusion'

        if checkpoint == 'stable_diffusion':
            self.text2image_model = self.text2image_model_stable
            if sampler == None: sampler = 'ddim'
        elif checkpoint == 'waifu_diffusion':
            self.text2image_model = self.text2image_model_waifu
            if sampler == None: sampler = 'k_euler_a'

        command_str = '/dream'
        command_str = command_str + f' prompt:{prompt} negative:{negative} checkpoint:{checkpoint} height:{str(height)} width:{width} guidance_scale:{guidance_scale} steps:{steps} sampler:{sampler} seed:{seed}'

        if init_image:
            command_str = command_str + f' init_image:{init_image.url}'

        if mask_image:
            command_str = command_str + f' mask_image:{mask_image.url}'

        if init_image or mask_image:
            if strength == None: strength = 0.75
            command_str = command_str + f' strength:{strength}'

        print(f'{command_str}')

        if self.dream_thread.is_alive():
            user_already_in_queue = False
            for queue_object in self.queue:
                if queue_object.ctx.author.id == ctx.author.id:
                    user_already_in_queue = True
                    break
            if user_already_in_queue:
                content=f'Please wait for your current image to finish generating before generating a new image'
                try:
                    await ctx.send_response(content=content, ephemeral=True)
                except:
                    await ctx.channel.send(content)
            else:
                self.queue.append(QueueObject(ctx, prompt, negative, height, width, guidance_scale, steps, seed,
                                              strength,
                                              init_image, mask_image, sampler, command_str))
                content=f'Dreaming for <@{ctx.author.id}> - Queue Position: ``{len(self.queue)}`` - ``{command_str}``'
                try:
                    await ctx.send_response(content=content)
                except:
                    await ctx.channel.send(content)
        else:
            await self.process_dream(QueueObject(ctx, prompt, negative, height, width, guidance_scale, steps, seed,
                                                 strength,
                                                 init_image, mask_image, sampler, command_str))
            content=f'Dreaming for <@{ctx.author.id}> - Queue Position: ``{len(self.queue)}`` - ``{command_str}``'
            try:
                await ctx.send_response(content=content)
            except:
                await ctx.channel.send(content)

    async def process_dream(self, queue_object: QueueObject):
        self.dream_thread = Thread(target=self.dream,
                                   args=(self.event_loop, queue_object))
        self.dream_thread.start()

    def dream(self, event_loop: AbstractEventLoop, queue_object: QueueObject):
        try:
            start_time = time.time()
            if (queue_object.init_image is None) and (queue_object.mask_image is None):
                samples, seed = self.text2image_model.dream(queue_object.prompt, queue_object.negative, queue_object.steps, False, False, 0.0,
                                                            1, 1, queue_object.guidance_scale, queue_object.seed,
                                                            queue_object.height, queue_object.width, False,
                                                            queue_object.sampler_name)
            elif queue_object.init_image is not None:
                image = Image.open(requests.get(queue_object.init_image.url, stream=True).raw).convert('RGB')
                samples, seed = self.text2image_model.translation(queue_object.prompt, queue_object.negative, image, queue_object.steps, 0.0,
                                                                  0,
                                                                  0, queue_object.guidance_scale,
                                                                  queue_object.strength, queue_object.seed,
                                                                  queue_object.height, queue_object.width,
                                                                  queue_object.sampler_name)
            else:
                image = Image.open(requests.get(queue_object.init_image.url, stream=True).raw).convert('RGB')
                mask = Image.open(requests.get(queue_object.mask_image.url, stream=True).raw).convert('RGB')
                samples, seed = self.text2image_model.inpaint(queue_object.prompt, queue_object.negative, image, mask, queue_object.steps, 0.0,
                                                              1, 1, queue_object.guidance_scale,
                                                              denoising_strength=queue_object.strength,
                                                              seed=queue_object.seed, height=queue_object.height,
                                                              width=queue_object.width,
                                                              sampler_name=queue_object.sampler_name)
            end_time = time.time()

            def upload_dream():
                async def run():
                    with BytesIO() as buffer:
                        samples[0].save(buffer, 'PNG')
                        buffer.seek(0)
                        embed = discord.Embed()
                        embed.colour = embed_color
                        embed.add_field(name='command', value=f'``{queue_object.command_str}``', inline=False)
                        embed.add_field(name='compute used', value='``{0:.3f}`` seconds'.format(end_time - start_time),
                                        inline=False)
                        embed.add_field(name='reactions', value='React with ❌ to delete your own generation\nReact with 🔁 to generate this picture again')
                        # fix errors if user doesn't have pfp
                        if queue_object.ctx.author.avatar is None:
                            embed.set_footer(
                                text=f'{queue_object.ctx.author.name}#{queue_object.ctx.author.discriminator}')
                        else:
                            embed.set_footer(
                                text=f'{queue_object.ctx.author.name}#{queue_object.ctx.author.discriminator}',
                                icon_url=queue_object.ctx.author.avatar.url)

                        event_loop.create_task(
                            queue_object.ctx.channel.send(content=f'<@{queue_object.ctx.author.id}>', embed=embed,
                                                        file=discord.File(fp=buffer, filename=f'{seed}.png')))
                asyncio.run(run())
            Thread(target=upload_dream, daemon=True).start()

        except Exception as e:
            embed = discord.Embed(title='txt2img failed', description=f'{e}\n{traceback.print_exc()}',
                                  color=embed_color)
            event_loop.create_task(queue_object.ctx.channel.send(embed=embed))
        if self.queue:
            # event_loop.create_task(self.process_dream(self.queue.pop(0)))
            self.dream(event_loop, self.queue.pop(0))

def setup(bot):
    src.bot.shanghai._stableCog = StableCog(bot)
    bot.add_cog(src.bot.shanghai._stableCog)
