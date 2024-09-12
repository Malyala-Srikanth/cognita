from backend.modules.dataloaders.githubloader import GithubLoader
from backend.modules.dataloaders.loader import register_dataloader
from backend.modules.dataloaders.localdirloader import LocalDirLoader
from backend.modules.dataloaders.webloader import WebLoader
from backend.modules.dataloaders.s3loader import S3Loader
from backend.settings import settings

register_dataloader("localdir", LocalDirLoader)
register_dataloader("web", WebLoader)
register_dataloader("github", GithubLoader)
register_dataloader("s3", S3Loader)
if settings.TFY_API_KEY:
    from backend.modules.dataloaders.truefoundryloader import TrueFoundryLoader

    register_dataloader("truefoundry", TrueFoundryLoader)
if settings.CARBON_AI_API_KEY:
    from backend.modules.dataloaders.carbondataloader import CarbonDataLoader

    register_dataloader("carbon", CarbonDataLoader)
