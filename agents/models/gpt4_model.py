from agents.model import Model, ModelInput, RoleType
from utils.logger_utils import LoggerUtils
import utils.constants as constants

import openai

import os

class GPT4_Model(Model):
	
	def __init__(self, is_debater: bool = True):
		self.is_debater = is_debater
		openai.organization = "INSERT_ORG_ID_HERE"
		openai.api_key = os.getenv("OPENAI_API_KEY")

	

