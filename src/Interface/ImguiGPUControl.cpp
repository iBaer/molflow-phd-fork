/*
Program:     MolFlow+ / Synrad+
Description: Monte Carlo simulator for ultra-high vacuum and synchrotron radiation
Authors:     Jean-Luc PONS / Roberto KERSEVAN / Marton ADY / Pascal BAEHR
Copyright:   E.S.R.F / CERN
Website:     https://cern.ch/molflow

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Full license text: https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html
*/

#if defined(MOLFLOW) and defined(GPUCOMPABILITY)

#include "ImguiGPUControl.h"
#include "imgui/imgui.h"
#include <imgui/imgui_internal.h>

#include "../src/GPUSim/GPUSettings.h"

extern MolFlow *mApp;

void ShowGPUWindow(MolFlow *mApp, bool *show_gpu, std::shared_ptr<flowgpu::MolflowGPUSettings> settings, bool *settingsChanged) {

    ImGui::PushStyleVar(
            ImGuiStyleVar_WindowMinSize,
            ImVec2(800.f, 0)); // Lift normal size constraint, however the presence of
    // a menu-bar will give us the minimum height we want.

    ImGui::Begin(
            "GPU Control", show_gpu,
            ImGuiWindowFlags_NoSavedSettings); // Pass a pointer to our bool
    // variable (the window will have
    // a closing button that will
    // clear the bool when clicked)
    ImGui::PopStyleVar(1);

    static flowgpu::MolflowGPUSettings gpu_settings = *settings;

    static bool simChanged = false;

    static int vec3i[3] = { static_cast<int>(gpu_settings.kernelDimensions[0]), static_cast<int>(gpu_settings.kernelDimensions[1]) , 1 };
    simChanged |= ImGui::Checkbox(
            "Allow desorption of particles",
            reinterpret_cast<bool *>(&gpu_settings.allowNewParticles));
    simChanged |= ImGui::InputInt3("NB Threads", vec3i);
    gpu_settings.kernelDimensions[0] = vec3i[0];
    gpu_settings.kernelDimensions[1] = vec3i[1] * vec3i[2];
    simChanged |= ImGui::Checkbox(
            "Batched random numbers",
            reinterpret_cast<bool *>(&gpu_settings.randomNumberMethod)); // Edit bools storing our window
    simChanged |= ImGui::InputInt("RNG cycles",reinterpret_cast<int *>( &gpu_settings.cyclesRNG));
    simChanged |= ImGui::InputInt("Recursive depth",reinterpret_cast<int *>( &gpu_settings.recursiveMaxDepth));
    simChanged |= ImGui::SliderFloat("Offset Magnitude Normal", &gpu_settings.offsetMagnitudeN, 0.00f, 0.0f, "%g");
    simChanged |= ImGui::SliderFloat("Offset Magnitude Center", &gpu_settings.offsetMagnitude, 0.00f, 0.0f, "%g");

    simChanged |= ImGui::SliderFloat("Gas molecular mass (g/mol)", &gpu_settings.gasMass, 0.00f, 0.0f, "%g");
    simChanged |= ImGui::Checkbox(
            "Maxwell Boltzmann Distribution",
            reinterpret_cast<bool *>(&gpu_settings.useMaxwellDistribution));
    if(simChanged) {
        *settings = gpu_settings;
        simChanged = false;
        *settingsChanged = true;
    }
}

#endif