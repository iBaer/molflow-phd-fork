//
// Created by Pascal Baehr on 04.11.21.
//

#include "CSVExporter.h"
#include "Buffer_shared.h"
#include "GeometrySimu.h"
#include <Helper/MathTools.h>
#include <cfloat> // DBL_EPSILON
#include <sstream>
#include <fstream>

/**
 * \brief Table for quick lookup of corresponding strings via enum values @see
 * FDetail
 */
static std::unordered_map<FDetail, std::string> const tableDetail = {
        {FDetail::F_ID,            "#"},
        {FDetail::F_STICKING,      "Sticking"},
        {FDetail::F_OPACITY,       "Opacity"},
        {FDetail::F_STRUCTURE,     "Structure"},
        {FDetail::F_LINK,          "Link"},
        {FDetail::F_DESORPTION,    "Desorption"},
        {FDetail::F_REFLECTION,    "Reflection"},
        {FDetail::F_TWOSIDED,      "2 Sided"},
        {FDetail::F_VERTEX,        "Vertex"},
        {FDetail::F_AREA,          "Area"},
        {FDetail::F_TEMP,          "Temperature 	(K)"},
        {FDetail::F_2DBOX,         "Facet 2D Box	"},
        {FDetail::F_TEXTURE_UV,    "Texture (u,v	)"},
        {FDetail::F_MESHSAMPLEPCM, "Mesh sample/	cm"},
        {FDetail::F_COUNT,         "Count"},
        {FDetail::F_MEMORY,        "Memory"},
        {FDetail::F_PLANARITY,     "Planarity"},
        {FDetail::F_PROFILE,       "Profile"},
        {FDetail::F_IMPINGEMENT,   "Imping.rate"},
        {FDetail::F_DENSITY1P,     "Density [1/m	3]"},
        {FDetail::F_DENSITYKGP,    "Density [kg/	m3]"},
        {FDetail::F_PRESSURE,      "Pressure [mb	ar]"},
        {FDetail::F_AVGSPEED,      "Av.mol.speed	[m/s]"},
        {FDetail::F_MCHITS,        "MC Hits"},
        {FDetail::F_EQUIVHITS,     "Equiv.hits"},
        {FDetail::F_NDESORPTIONS,  "Des."},
        {FDetail::F_EQUIVABS,      "Equiv.abs."}};

static const char *desStr[] = {"None", "Uniform", "Cosine", "Cosine^"};

static const char *profStr[] = {
        "None", "Pressure (\201)", "Pressure (\202)", "Angular",
        "Speed distr.", "Ort. velocity", "Tan. velocity"};

static const char *ynStr[] = {"No", "Yes"};

double GetMoleculesPerTP(size_t moment, SimulationModel *model,
                         GlobalSimuState *glob) {
    if (glob->globalHits.globalHits.nbDesorbed == 0)
        return 0; // avoid division by 0
    if (moment == 0) {
        // Constant flow
        // Each test particle represents a certain real molecule influx per second
        return model->wp.finalOutgassingRate /
                (double)glob->globalHits.globalHits.nbDesorbed;
    } else {
        // Time-dependent mode
        // Each test particle represents a certain absolute number of real
        // molecules. Since Molflow displays per-second values (imp.rate, etc.), the
        // sampled time window length is only a fraction of a second. For example,
        // if dt=0.1s, we have collected only 1/10th of what would happen during a
        // second. Hence we DIVIDE by the time window length, even if it's
        // uninuitional.
        auto timeWindow =
                model->tdParams.moments[moment - 1].second -
                model->tdParams.moments[moment - 1]
                        .first; // TODO: Can we get access to the time windows directly?
        return (model->wp.totalDesorbedMolecules / timeWindow) /
                (double)glob->globalHits.globalHits.nbDesorbed;
    }
}

/**
 * \brief Function that calculates a density correction factor [0..1] (with 1.0
 * = no correction) \return correction factor value [0..1]
 */
double DensityCorrection(const FacetHitBuffer &fHit) {
    // Correction for double-density effect (measuring density on
    // desorbing/absorbing facets):

    // Normally a facet only sees half of the particles (those moving towards it).
    // So it multiplies the "seen" density by two. However, in case of desorption
    // or sticking, the real density is not twice the "seen" density, but a bit
    // less, therefore this reduction factor If only desorption, or only
    // absorption, the correction factor is 0.5, if no des/abs, it's 1.0, and in
    // between, see below

    if (fHit.nbMCHit > 0 || fHit.nbDesorbed > 0) {
        if (fHit.nbAbsEquiv > 0.0 ||
            fHit.nbDesorbed > 0) { // otherwise save calculation time
            return 1.0 - (fHit.nbAbsEquiv + (double) fHit.nbDesorbed) /
                         (fHit.nbHitEquiv + (double) fHit.nbDesorbed) / 2.0;
        } else
            return 1.0;
    } else
        return 1.0;
}

double GetArea(const SubprocessFacet &fac) {
    return fac.sh.area * (fac.sh.is2sided ? 2.0 : 1.0);
}

/**
 * \brief Gives a string which counts values corresponding to the facet settings
 * \param f Pointer to a facet
 * \return char pointer taking a string with the count value(s)
 */
char *GetCountStr(SubprocessFacet *f) {
    static char ret[128];
    strcpy(ret, "");
    if (f->sh.countDes)
        strcat(ret, "DES");
    if (f->sh.countAbs) {
        if (strlen(ret) == 0) {
            strcat(ret, "ABS");
        } else {
            strcat(ret, "+ABS");
        }
    }
    if (f->sh.countRefl) {
        if (strlen(ret) == 0) {
            strcat(ret, "REFL");
        } else {
            strcat(ret, "+REFL");
        }
    }
    if (f->sh.countTrans) {
        if (strlen(ret) == 0) {
            strcat(ret, "TRANS");
        } else {
            strcat(ret, "+TRANS");
        }
    }
    return ret;
}

/**
 * \brief Prints table values inside the corresponding cell
 * \param idx Facet ID (local for table)
 * \param f Pointer to a facet
 * \param mode which kind of value has to be evaluated and printed
 * \return char pointer taking a string with the count value(s)
 */
char *CSVExporter::FormatCell(FDetail mode, size_t idx, GlobalSimuState *glob,
                              SimulationModel *model) {
    static char ret[1024];
    strcpy(ret, "");

    // Maybe validate globsimustate/model sanity (same nb facets etc) before
    if(model->facets.size() <= idx || glob->facetStates.size() <= idx)
        return ret;

    auto *facet = model->facets[idx].get();
    auto moment = 0;
    auto &fHit = glob->facetStates[idx].momentResults[moment].hits;

    switch (mode) {
        case FDetail::F_ID:
            sprintf(ret, "%zd", idx + 1);
            break;
        case FDetail::F_STICKING:
            sprintf(ret, "%g", facet->sh.sticking);
            break;
        case FDetail::F_OPACITY:
            sprintf(ret, "%g", facet->sh.opacity);
            break;
        case FDetail::F_STRUCTURE: {
            std::ostringstream out;
            if (facet->sh.superIdx == -1)
                out << "All";
            else
                out << (facet->sh.superIdx + 1);
            sprintf(ret, "%s", out.str().c_str());
            break;
        }
        case FDetail::F_LINK:
            sprintf(ret, "%zd", facet->sh.superDest);
            break;
        case FDetail::F_DESORPTION:
            if (facet->sh.desorbType == DES_COSINE_N) {
                sprintf(ret, "%s%g", desStr[facet->sh.desorbType],
                        facet->sh.desorbTypeN); // append exponent
            } else {
                sprintf(ret, "%s", desStr[facet->sh.desorbType]);
            }
            break;
        case FDetail::F_REFLECTION:
            sprintf(ret, "%g diff. %g spec. %g cos^%g",
                    facet->sh.reflection.diffusePart, facet->sh.reflection.specularPart,
                    1.0 - facet->sh.reflection.diffusePart -
                    facet->sh.reflection.specularPart,
                    facet->sh.reflection.cosineExponent);
            break;
        case FDetail::F_TWOSIDED:
            sprintf(ret, "%s", ynStr[facet->sh.is2sided]);
            break;
        case FDetail::F_VERTEX:
            sprintf(ret, "%zd", facet->sh.nbIndex);
            break;
        case FDetail::F_AREA:
            if (facet->sh.is2sided)
                sprintf(ret, "2*%g", facet->sh.area);
            else
                sprintf(ret, "%g", facet->sh.area);
            break;
        case FDetail::F_TEMP:
            sprintf(ret, "%g", facet->sh.temperature);
            break;
        case FDetail::F_2DBOX:
            sprintf(ret, "%g x %g", facet->sh.U.Norme(), facet->sh.V.Norme());
            break;
        case FDetail::F_TEXTURE_UV:
            if (facet->sh.isTextured) {
                sprintf(ret, "%zdx%zd (%g x %g)", facet->sh.texWidth, facet->sh.texHeight,
                        facet->sh.texWidth_precise, facet->sh.texHeight_precise);
            } else {
                sprintf(ret, "None");
            }
            break;
        case FDetail::F_MESHSAMPLEPCM: {
            double tRatioU, tRatioV;
            const double nU = facet->sh.U.Norme();
            const double nV = facet->sh.V.Norme();

            tRatioU = facet->sh.texWidth_precise / nU;
            tRatioV = facet->sh.texHeight_precise / nV;

            if (std::abs(tRatioU - tRatioV) <= DBL_EPSILON) {
                tRatioV = tRatioU;
            }

            if (IsEqual(tRatioU, tRatioV))
                sprintf(ret, "%g", tRatioU);
            else
                sprintf(ret, "%g x %g", tRatioU, tRatioV);
            break;
        }
        case FDetail::F_COUNT:
            sprintf(ret, "%s", GetCountStr(facet));
            break;
        case FDetail::F_MEMORY:
            sprintf(ret, "%s", "N/A");
            // sprintf(ret, "%s", FormatMemory(facet->GetTexRamSize(1 +
            // worker->moments.size())));
            break;
        case FDetail::F_PLANARITY: {
            // Facet planarity
            Vector3d p0 = model->vertices3[facet->indices[0]];
            double A = facet->sh.N.x;
            double B = facet->sh.N.y;
            double C = facet->sh.N.z;
            double D = -Dot(facet->sh.N, p0);

            double planarityError = 0.0;
            for (size_t i = 3; i < facet->sh.nbIndex;
                 i++) { // First 3 vertices are by def on a plane
                const Vector3d &p = model->vertices3[facet->indices[i]];
                double d = A * p.x + B * p.y + C * p.z + D;
                planarityError = std::max(abs(d), planarityError);
            }
            sprintf(ret, "%facet", planarityError);
            break;
        }
        case FDetail::F_PROFILE:
            sprintf(ret, "%s", profStr[facet->sh.profileType]);
            break;
        case FDetail::F_IMPINGEMENT: // imp.rate
        {
            double dCoef =
                    1E4 * GetMoleculesPerTP(
                            moment, model,
                            glob); // 1E4 is conversion from m2 to cm2; 0.01 is Pa->mbar
            sprintf(ret, "%g", fHit.nbHitEquiv / GetArea(*facet) * dCoef);
            // 11.77=sqrt(8*8.31*293.15/3.14/0.028)/4/10
            break;
        }
        case FDetail::F_DENSITY1P: // particle density
        {
            double dCoef =
                    1E4 * GetMoleculesPerTP(moment, model, glob) *
                    DensityCorrection(
                            fHit); // 1E4 is conversion from m2 to cm2; 0.01 is Pa->mbar

            sprintf(ret, "%g", fHit.sum_1_per_ort_velocity / GetArea(*facet) * dCoef);

            break;
        }
        case FDetail::F_DENSITYKGP: // gas density
        {
            double dCoef =
                    1E4 * GetMoleculesPerTP(moment, model, glob) *
                    DensityCorrection(
                            fHit); // 1E4 is conversion from m2 to cm2; 0.01 is Pa->mbar

            sprintf(ret, "%g",
                    fHit.sum_1_per_ort_velocity / GetArea(*facet) * dCoef *
                    model->wp.gasMass / 1000.0 / 6E23);
            break;
        }
        case FDetail::F_PRESSURE: // avg.pressure
        {
            double dCoef = 1E4 * GetMoleculesPerTP(moment, model, glob) *
                           (model->wp.gasMass / 1000 / 6E23) *
                           0.0100; // 1E4 is conversion from m2 to cm2; 0.01 is Pa->mbar

            sprintf(ret, "%g", fHit.sum_v_ort * dCoef / GetArea(*facet));
            break;
        }
        case FDetail::F_AVGSPEED: // avg. gas speed (estimate)
            /*sprintf(ret, "%g", 4.0*(double)(fHit.hit.nbMCHit+fHit.hit.nbDesorbed) /
             * fHit.hit.sum_1_per_ort_velocity);*/
            sprintf(ret, "%g",
                    (fHit.nbHitEquiv + static_cast<double>(fHit.nbDesorbed)) /
                    fHit.sum_1_per_velocity);
            //<v_surf>=2*<v_surFDetail::F_ort>
            //<v_gas>=1/<1/v_surf>
            break;
        case FDetail::F_MCHITS:
            sprintf(ret, "%zd", fHit.nbMCHit);
            break;
        case FDetail::F_EQUIVHITS:
            sprintf(ret, "%g", fHit.nbHitEquiv);
            break;
        case FDetail::F_NDESORPTIONS:
            sprintf(ret, "%zd", fHit.nbDesorbed);
            break;
        case FDetail::F_EQUIVABS:
            sprintf(ret, "%g", fHit.nbAbsEquiv);
            break;
    }

    return ret;
}

std::string CSVExporter::GetHeader(const std::vector<FDetail> &selectedValues) {
    std::string buffer;
    for (auto &mode: selectedValues) {
        buffer.append(tableDetail.at(mode));
        buffer.append(",");
    }
    if(!selectedValues.empty() && !buffer.empty())
        buffer.pop_back(); // remove last delimiter
    return buffer;
}

std::string
CSVExporter::GetLineForFacet(size_t idx,
                             const std::vector<FDetail> &selectedValues,
                             GlobalSimuState *glob, SimulationModel *model) {
    std::string buffer;
    for (auto &mode: selectedValues) {
        buffer.append(FormatCell(mode, idx, glob, model));
        buffer.append(",");
    }

    return buffer;
}

std::string CSVExporter::GetFacetDetailsCSV(GlobalSimuState *glob,
                                            SimulationModel *model) {
    // Generate list of all modes
    std::vector<FDetail> selectedValues;
    selectedValues.reserve(tableDetail.size());
    for (auto &entry: tableDetail) {
        selectedValues.push_back(entry.first);
    }

    std::string buffer;
    buffer.append(GetHeader(selectedValues));
    for (int idx = 0; idx < model->facets.size(); ++idx) {
        buffer.append(GetLineForFacet(idx, selectedValues, glob, model));
        buffer.append("\n");
    }

    return buffer;
}

int CSVExporter::ExportAllFacetDetails(const std::string& fileName, GlobalSimuState *glob,
                                       SimulationModel *model) {

    std::string facDetails = CSVExporter::GetFacetDetailsCSV(glob, model);

    std::ofstream ofs(fileName);
    ofs << facDetails;
    ofs.close();

    return 0;
}