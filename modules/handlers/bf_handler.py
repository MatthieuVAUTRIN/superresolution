from modules.beamformers.DAS import Das
from modules.beamformers.DAS_gpu import DasGPU
from modules.beamformers.gDMAS import gDMAS
from modules.beamformers.GCF_NSI import GcfNSI
from modules.beamformers.DMAS import Dmas


class BeamformerHandler(Das, DasGPU, Dmas, gDMAS, GcfNSI):
    def __init__(
        self,
        beamformer_type: str,
        params,
        sampling_rate,
        samples,
        using_real_data,
        interpolate_data,
        nsi_dc_offset,
        bf_dtype,
        debug,
    ):
        self.beamformer_type = beamformer_type.lower()
        self.nsi_dc_offset = nsi_dc_offset

        self.params = params

        self.sampling_rate = sampling_rate
        self.samples = samples

        self.using_real_data = using_real_data
        self.interpolate_data = interpolate_data
        self.bf_dtype = bf_dtype

        self.debug = debug

        switcher = {
            "das": self._init_das,
            "gdmas": self._init_gdmas,
            "dmas": self._init_dmas,
            "gcf_nsi": self._init_gcf_nsi,
        }

        init_method = switcher.get(
            beamformer_type, lambda: "Invalid beamformer type"
        )  # setting default to a lambda function to match the callable type

        #! create wrapper later if needed bevause of paramters discrepancies
        if callable(init_method):
            init_method()
        else:
            raise ValueError(init_method)

    def _init_das(self):
        print("Initializing DAS beamformer - CPU and GPU versions")
        Das.__init__(
            self,
            params=self.params,
            nsi_dc_offset=self.nsi_dc_offset,
            sampling_rate=self.sampling_rate,
            samples=self.samples,
            using_real_data=self.using_real_data,
            interpolate_data=self.interpolate_data,
            bf_dtype=self.bf_dtype,
            debug=self.debug,
        )

        DasGPU.__init__(
            self,
            params=self.params,
            nsi_dc_offset=self.nsi_dc_offset,
            sampling_rate=self.sampling_rate,
            samples=self.samples,
            using_real_data=self.using_real_data,
            interpolate_data=self.interpolate_data,
            bf_dtype=self.bf_dtype,
            debug=self.debug,
        )

    def _init_dmas(self):
        print("Initializing DMAS beamformer - CPU version")
        Dmas.__init__(
            self,
            params=self.params,
            nsi_dc_offset=self.nsi_dc_offset,
            sampling_rate=self.sampling_rate,
            samples=self.samples,
            using_real_data=self.using_real_data,
            interpolate_data=self.interpolate_data,
            bf_dtype=self.bf_dtype,
            debug=self.debug,
        )

    def _init_gdmas(self):
        print("Initializing gDMAS beamformer - CPU version")
        gDMAS.__init__(
            self,
            params=self.params,
            nsi_dc_offset=self.nsi_dc_offset,
            sampling_rate=self.sampling_rate,
            samples=self.samples,
            using_real_data=self.using_simulation_data,
            interpolate_data=self.interpolate_data,
            bf_dtype=self.bf_dtype,
            debug=self.debug,
        )

    def _init_gcf_nsi(self):
        print("Initializing GCF_NSI beamformer - CPU version")
        GcfNSI.__init__(
            self,
            params=self.params,
            nsi_dc_offset=self.nsi_dc_offset,
            sampling_rate=self.sampling_rate,
            samples=self.samples,
            using_real_data=self.using_simulation_data,
            interpolate_data=self.interpolate_data,
            bf_dtype=self.bf_dtype,
            debug=self.debug,
        )
