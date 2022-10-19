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
                if message.embeds[0].fields[0].name == 'command':
                    await message.add_reaction('❌')
                    await message.add_reaction('🔁')
            except:
                pass

    async def on_raw_reaction_add(self, ctx):
        if ctx.emoji.name == '❌':
            message = await self.get_channel(ctx.channel_id).fetch_message(ctx.message_id)
            if message.embeds:
                # look at the message footer to see if the generation was by the user who reacted
                if message.embeds[0].footer.text == f'{ctx.member.name}#{ctx.member.discriminator}':
                    await message.delete()

        if ctx.emoji.name == '🔁':
            message = await self.get_channel(ctx.channel_id).fetch_message(ctx.message_id)
            if message.embeds and ctx.member != self.user:
                # try:
                    # Check if the message from Shanghai was actually a generation
                    if message.embeds[0].fields[0].name == 'command':
                        command = message.embeds[0].fields[0].value
                        # messageReference = await self.get_channel(ctx.channel_id).fetch_message(message.reference.message_id)
                        ctx.author = ctx.member
                        ctx.channel = self.get_channel(ctx.channel_id)

                        # ``/dream prompt:anime girl negative: checkpoint:waifu_diffusion height:512 width:512 guidance_scale:7.0 steps:20 sampler:k_euler_a seed:397814580``
                        await _stableCog.dream_handler(ctx=ctx,
                            prompt=self.find_between(command, '``/dream prompt:', ' negative:'),
                            negative=self.find_between(command, ' negative:', ' checkpoint:'),
                            checkpoint=self.find_between(command, ' checkpoint:', ' height:'),
                            height=int(self.find_between(command, ' height:', ' width:')),
                            width=int(self.find_between(command, ' width:', ' guidance_scale:')),
                            guidance_scale=float(self.find_between(command, ' guidance_scale:', ' steps:')),
                            steps=int(self.find_between(command, ' steps:', ' sampler:')),
                            sampler=self.find_between(command, ' sampler:', ' seed:'),
                            seed=-1
                        )
                        # cog.dream_handler
                # except:
                #     pass

    def find_between(self, s, first, last):
        try:
            start = s.index( first ) + len( first )
            end = s.index( last, start )
            return s[start:end]
        except ValueError:
            return ''
