//
// Created by pbahr on 22/06/2022.
//

#ifndef MOLFLOW_PROJ_GPUSETTINGS_H
#define MOLFLOW_PROJ_GPUSETTINGS_H

namespace flowgpu {
    //! Settings structure for GPU simulations that are in particular used on the CUDA kernel
    struct MolflowGPUSettings {
        MolflowGPUSettings() = default;

        float gasMass{28.0f};
        bool useMaxwellDistribution{true};
        size_t recursiveMaxDepth{0};
        size_t cyclesRNG{1};
        bool randomNumberMethod{true}; /*! 0=bulked, 1=ad hoc */
        bool allowNewParticles{true}; /*! 0=stop desorption, finish existing, 1=allow desorption */

        /*bool	 lowFluxMode;
        double	 lowFluxCutoff;*/

        unsigned int kernelDimensions[2]{1920,128}; // blocks and threads per block
        float offsetMagnitude{0.0f}; // adaptive offset towards center
        float offsetMagnitudeN{1.0f}; // adaptive offset towards normal
    };
}
#endif //MOLFLOW_PROJ_GPUSETTINGS_H
