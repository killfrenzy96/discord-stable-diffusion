import asyncio
import os
from abc import ABC

import discord
from discord.ext import commands
from src.core.logging import get_logger
# from src.bot.stablecog import _stableCog


class Shanghai(commands.Bot, ABC):
    def __init__(self, args):
        global _stableCog
        _stableCog = None

        intents = discord.Intents.default()
        intents.members = True
        super().__init__(command_prefix=args.prefix, intents=intents)
        self.args = args
        self.logger = get_logger(__name__)
        self.load_extension('src.bot.stablecog')

    async def on_ready(self):
        self.logger.info(f'Logged in as {self.user.name} ({self.user.id})')
        await self.change_presence(
            activity=discord.Activity(type=discord.ActivityType.watching, name='you over the seven seas.'))

    async def on_message(self, message):
        if message.author == self.user:
            try:
                # Check if the message from Shanghai was actually a generation
                # if message.embeds[0].fields[0].name == 'react':
                if message.content.startswith('<@') and 'Dreaming - Queue Position:' not in message.content:
                    await message.add_reaction('❌')
                    if '``/dream prompt:' in message.content:
                        await message.add_reaction('🔁')
            except:
                pass

    async def on_raw_reaction_add(self, ctx: discord.RawReactionActionEvent):
        if ctx.emoji.name == '❌':
            channel = self.get_channel(ctx.channel_id)
            if channel == None:
                channel = await self.fetch_channel(ctx.channel_id)

            message: discord.Message = await channel.fetch_message(ctx.message_id)

            author = message.author
            if author == None:
                return

            # message = await self.get_channel(ctx.channel_id).fetch_message(ctx.message_id)
            # if message.embeds:
                # look at the message footer to see if the generation was by the user who reacted
                # if message.embeds[0].footer.text == f'{ctx.member.name}#{ctx.member.discriminator}':
                #     await message.delete()

            if author.id == self.user.id and message.content.startswith(f'<@{ctx.user_id}>'):
                await message.delete()

        if ctx.emoji.name == '🔁':
            channel = self.get_channel(ctx.channel_id)
            if channel == None:
                channel = await self.fetch_channel(ctx.channel_id)

            message: discord.Message = await channel.fetch_message(ctx.message_id)

            user = ctx.member
            if user == None:
                user = await self.fetch_user(ctx.user_id)

            # message = await self.get_channel(ctx.channel_id).fetch_message(ctx.message_id)

            if message.author.id == self.user.id and user.id != self.user.id:
                # try:
                    # Check if the message from Shanghai was actually a generation
                    # if message.embeds[0].fields[0].name == 'command':
                    if '``/dream prompt:' in message.content:
                        # command = message.embeds[0].fields[0].value
                        command = '``/dream ' + self.sh_find_between(message.content, '``/dream ', '``') + '``'
                        # messageReference = await self.get_channel(ctx.channel_id).fetch_message(message.reference.message_id)

                        message.author = user

                        class image_url:
                            url: str

                        prompt = self.sh_get_param(command, 'prompt')

                        negative = self.sh_get_param(command, 'negative')
                        if negative == '-': negative = ''

                        checkpoint = self.sh_get_param(command, 'checkpoint')
                        if checkpoint == '': checkpoint = 'stable_diffusion'

                        try:
                            height = int(self.sh_get_param(command, 'height'))
                        except:
                            height = 512

                        try:
                            width = int(self.sh_get_param(command, 'width'))
                        except:
                            width = 512

                        try:
                            guidance_scale = float(self.sh_get_param(command, 'guidance_scale'))
                        except:
                            guidance_scale = 7.0

                        try:
                            step = int(self.sh_get_param(command, 'steps'))
                        except:
                            step = 20

                        try:
                            sampler = self.sh_get_param(command, 'sampler')
                        except:
                            sampler = 'ddim'

                        seed = -1

                        try:
                            strength = float(self.sh_get_param(command, 'strength'))
                        except:
                            strength = 0.75

                        try:
                            batch = int(self.sh_get_param(command, 'batch'))
                        except:
                            batch = 1

                        init_image = image_url()
                        mask_image = image_url()
                        init_image.url = self.sh_get_param_url(command, 'init_image')
                        mask_image.url = self.sh_get_param_url(command, 'mask_image')
                        if init_image.url == '': init_image = None
                        if mask_image.url == '': mask_image = None

                        await _stableCog.dream_handler(ctx=message,
                            prompt=prompt,
                            negative=negative,
                            checkpoint=checkpoint,
                            height=height,
                            width=width,
                            guidance_scale=guidance_scale,
                            steps=step,
                            sampler=sampler,
                            seed=seed,
                            init_image=init_image,
                            mask_image=mask_image,
                            strength=strength,
                            batch=batch
                        )
                        # cog.dream_handler
                # except:
                #     pass

    def sh_get_param_url(self, command, param):
        return self.sh_find_between(command, f'{param}:', ' ')

    def sh_get_param(self, command, param):
        result = self.sh_find_between(command, f'{param}:', ':')
        if result == '':
            result = self.sh_find_between(command, f'{param}:', '``')
        else:
            result = result.rsplit(' ', 1)[0]

        return result

    def sh_find_between(self, s, first, last):
        try:
            start = s.index( first ) + len( first )
            end = s.index( last, start )
            return s[start:end]
        except ValueError:
            return ''
