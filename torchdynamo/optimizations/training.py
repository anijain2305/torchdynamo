import logging

import torch

from torchdynamo.utils import clone_inputs
from torchdynamo.utils import count_calls
from torchdynamo.utils import counters

from .analysis import has_mutation
from .backends import BACKENDS
from .normalize import normalize_ir

log = logging.getLogger(__name__)

counters["aot_autograd"]["total"] = 0
counters["aot_autograd"]["not_ok"] = 0
counters["aot_autograd"]["ok"] = 0

class AOTAutogradStrategy(object):
    """Base class for backend strategies that use AOT Autograd"""

    @classmethod
    def compile_fn(cls, gm: torch.fx.GraphModule, example_inputs):
        if count_calls(gm.graph) < 2:
            return gm.forward  # no point for tiny graphs
        return cls(gm, example_inputs).verified_candidate()

    def __init__(self, gm: torch.fx.GraphModule, example_inputs):
        super(AOTAutogradStrategy, self).__init__()
        counters["aot_autograd"]["total"] += 1
        self.use_fallback = False
        self.original_example_inputs = example_inputs
        try:
            self.gm = normalize_ir(gm, self.example_inputs)
        except Exception:
            raise
            log.debug("TorchDynamo unable to remove mutation")
            self.gm = gm
            self.use_fallback = True
            pass

        gm_inputs = list(filter(lambda x: x.op == "placeholder", gm.graph.nodes))

        # TODO - AOT Autograd has some know issues. Here, we check for those and
        # use fallback when necessary.
        # 1) gather_backward (pytorch_struct) - https://github.com/pytorch/functorch/issues/591
        for node in self.gm.graph.nodes:
            if node.target == torch.gather:
                log.debug(
                    "Graph has gather op. AOT Autograd does not handle gather correctly. Using fallback."
                )
                self.use_fallback = True

        # 2) LSTM module (tts_angular) - https://github.com/pytorch/functorch/issues/586
        for submod in self.gm.modules():
            if submod.__class__.__name__ == "LSTM":
                self.use_fallback = True

        has_param_as_input = False
        for ex in example_inputs:
            if isinstance(ex, torch.nn.parameter.Parameter):
                has_param_as_input = True

        if (
            has_mutation(self.gm, self.example_inputs)
            or len(gm_inputs) == 0
            or has_param_as_input
        ):
            self.use_fallback = True

    @property
    def example_inputs(self):
        return clone_inputs(self.original_example_inputs)

    def verified_candidate(self):
        if self.use_fallback:
            log.debug("Unable to use AOT Autograd because graph has mutation")
            counters["aot_autograd"]["not_ok"] += 1
            return self.gm
        cg = self.candidate()
        if cg is None:
            counters["aot_autograd"]["not_ok"] += 1
            raise RuntimeError("AOT Autograd failed to compile")
        counters["aot_autograd"]["ok"] += 1
        return cg

    def candidate(self):
        raise NotImplementedError()


class AOTAutogradEagerStrategy(AOTAutogradStrategy):
    """Useful for debugging purpose"""

    def candidate(self):
        from functorch.compile import nop

        return BACKENDS["aot_autograd"](
            self.gm, self.example_inputs, fw_compiler=nop
        )


aot_autograd_debug_strategy1 = AOTAutogradEagerStrategy.compile_fn


class AOTAutogradNNCStrategy(AOTAutogradStrategy):

    def candidate(self):
        from functorch.compile import ts_compile

        return BACKENDS["aot_autograd"](
            self.gm, self.example_inputs, fw_compiler=ts_compile
        )


aot_autograd_nnc_strategy = AOTAutogradNNCStrategy.compile_fn

# Global counter to differentiate between different graphs.
graph_idx = 0


class AOTAutogradEagerSaveStrategy(AOTAutogradEagerStrategy):
    """Saves all the gm models so that we can run them separately"""

    def candidate(self):
        global graph_idx
        module_idx = "module_" + str(graph_idx)
        self.gm.to_folder(module_idx, "Bar")
        for idx, x in enumerate(self.example_inputs):
            torch.save(x, module_idx + "_tensor" + str(idx) + ".pt")
        graph_idx += 1
        return super(AOTAutogradEagerSaveStrategy, self).candidate()


aot_autograd_debug_strategy2 = AOTAutogradEagerSaveStrategy.compile_fn


class AOTAutogradMemoryEfficientFusion(AOTAutogradStrategy):
    """Use Min cut rematerilization and NVFuser with AOT Autograd"""

    def candidate(self):
        return BACKENDS["aot_autograd"](self.gm, self.example_inputs)


aot_autograd_speedup_strategy = AOTAutogradMemoryEfficientFusion.compile_fn
