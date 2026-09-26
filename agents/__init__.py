from agents.bc import BCAgent
from agents.cdp import CDPAgent
from agents.fbrac import FBRACAgent
from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent

agents = {
    'bc': BCAgent,
    'cdp': CDPAgent,
    'fbrac': FBRACAgent,
    'fql': FQLAgent,
    'ifql': IFQLAgent,
    'iql': IQLAgent,
    'rebrac': ReBRACAgent,
}
