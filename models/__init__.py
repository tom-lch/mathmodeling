from .adaboost import adaboost, adaboost_single
from .rdf import dtreec, rforest, nbbnb, nbgnb, nknc, nbrnc, dtreec_single, rforest_single, nbbnb_single, nbgnb_single, nknc_single, nbrnc_single
from .svm import svm, svm_single
from .nn import nnmlpc, nnmlpc_single

ms = [adaboost, dtreec, rforest, svm, nbbnb, nbgnb, nknc, nnmlpc, nbrnc,svm_single, nnmlpc_single,
        adaboost_single, dtreec_single, rforest_single, nbbnb_single, nbgnb_single, nknc_single, nbrnc_single]