from neural_network.optimizers.initializers import Initializers
import logging

from utils.logs import create_logger
log = create_logger(__name__)

init = Initializers()
log.info(init.xavier(dim0=9,dim1=6, normalized=True))
log.info(init.heuristic(lower=-1, upper=1, dim0=9, dim1=6))
log.info(init.he(lower=-1, upper=1, dim0=9, dim1=6))
optimizer_param=dict(name='heuristic', lower=-1, upper=1, dim0=9, dim1=6)
print(optimizer_param)
intializer=getattr(Initializers(),'he')
print(intializer(**optimizer_param))
