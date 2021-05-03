//
// Created by Pascal Baehr on 28.04.20.
//

#include <sstream>
#include <Helper/MathTools.h>
#include <cmath>
#include <set>
#include <Simulation/CDFGeneration.h>
#include <Simulation/IDGeneration.h>
#include "GeometrySimu.h"
#include "IntersectAABB_shared.h" // include needed for recursive delete of AABBNODE

SuperStructure::SuperStructure()
{
    //aabbTree = NULL;
}

SuperStructure::~SuperStructure()
{
    aabbTree.reset();
}

bool SubprocessFacet::InitializeOnLoad(const size_t &id, const size_t &nbMoments) {
    globalId = id;
    //ResizeCounter(nbMoments); //Initialize counter
    if (!InitializeLinkAndVolatile(id)) return false;
    InitializeOutgassingMap();
    InitializeAngleMap();
    InitializeTexture(nbMoments);
    InitializeProfile(nbMoments);
    InitializeDirectionTexture(nbMoments);
    InitializeHistogram(nbMoments);

    return true;
}

size_t SubprocessFacet::InitializeHistogram(const size_t &nbMoments) const
{
    //FacetHistogramBuffer hist;
    //hist.Resize(sh.facetHistogramParams);
    //tmpHistograms = std::vector<FacetHistogramBuffer>(1 + nbMoments, hist);
    size_t histSize = (1 + nbMoments) *
                                   (sh.facetHistogramParams.GetBouncesDataSize()
                                    + sh.facetHistogramParams.GetDistanceDataSize()
                                    + sh.facetHistogramParams.GetTimeDataSize());

    return histSize;
}

size_t SubprocessFacet::InitializeDirectionTexture(const size_t &nbMoments)
{
    size_t directionSize = 0;

        //Direction
    if (sh.countDirection) {
        directionSize = sh.texWidth*sh.texHeight * sizeof(DirectionCell);
        try {
            //direction = std::vector<std::vector<DirectionCell>>(1 + nbMoments, std::vector<DirectionCell>(sh.texWidth*sh.texHeight));
        }
        catch (...) {
            throw std::runtime_error("Not enough memory to load direction textures");
        }
    }
    else
        directionSize = 0;
    return directionSize;
}

size_t SubprocessFacet::InitializeProfile(const size_t &nbMoments)
{    size_t profileSize = 0;

    //Profiles
    if (sh.isProfile) {
        profileSize = PROFILE_SIZE * sizeof(ProfileSlice);
        try {
            //profile = std::vector<std::vector<ProfileSlice>>(1 + nbMoments, std::vector<ProfileSlice>(PROFILE_SIZE));
        }
        catch (...) {
            throw std::runtime_error("Not enough memory to load profiles");
            return false;
        }
    }
    else
        profileSize = 0;
    return profileSize;
}

size_t SubprocessFacet::InitializeTexture(const size_t &nbMoments)
{
    size_t textureSize = 0;
    //Textures
    if (sh.isTextured) {
        size_t nbE = sh.texWidth*sh.texHeight;
        largeEnough.resize(nbE);
        textureSize = nbE * sizeof(TextureCell);
        /*try {
            texture = std::vector<std::vector<TextureCell>>(1 + nbMoments, std::vector<TextureCell>(nbE));
        }
        catch (...) {
            throw std::runtime_error("Not enough memory to load textures");
            return false;
        }*/
        // Texture increment of a full texture element
        double fullSizeInc = (sh.texWidthD * sh.texHeightD) / (sh.U.Norme() * sh.V.Norme());
        for (size_t j = 0; j < nbE; j++) { //second pass, filter out very small cells
            largeEnough[j] = textureCellIncrements[j] < (5.0*fullSizeInc);
        }

        //double iw = 1.0 / (double)sh.texWidthD;
        //double ih = 1.0 / (double)sh.texHeightD;
        //double rw = sh.U.Norme() * iw;
        //double rh = sh.V.Norme() * ih;
    }
    else
        textureSize = 0;
    return textureSize;
}


size_t SubprocessFacet::InitializeAngleMap()
{
    //Incident angle map
    size_t angleMapSize = 0;
    if (sh.desorbType == DES_ANGLEMAP) { //Use mode
        //if (angleMapCache.empty()) throw Error(("Facet " + std::to_string(globalId + 1) + ": should generate by angle map but has none recorded.").c_str());

        //Construct CDFs
        try {
            angleMap.phi_CDFsums.resize(sh.anglemapParams.thetaLowerRes + sh.anglemapParams.thetaHigherRes);
        }
        catch (...) {
            throw std::runtime_error("Not enough memory to load incident angle map (phi CDF line sums)");
            return false;
        }
        try {
            angleMap.theta_CDF.resize(sh.anglemapParams.thetaLowerRes + sh.anglemapParams.thetaHigherRes);
        }
        catch (...) {
            throw std::runtime_error("Not enough memory to load incident angle map (line sums, CDF)");
            return false;
        }
        try {
            angleMap.phi_CDFs.resize(sh.anglemapParams.phiWidth * (sh.anglemapParams.thetaLowerRes + sh.anglemapParams.thetaHigherRes));
        }
        catch (...) {
            throw std::runtime_error("Not enough memory to load incident angle map (CDF)");
            return false;
        }

        //First pass: determine sums
        angleMap.theta_CDFsum = 0;
        memset(angleMap.phi_CDFsums.data(), 0, sizeof(size_t) * (sh.anglemapParams.thetaLowerRes + sh.anglemapParams.thetaHigherRes));
        for (size_t thetaIndex = 0; thetaIndex < (sh.anglemapParams.thetaLowerRes + sh.anglemapParams.thetaHigherRes); thetaIndex++) {
            for (size_t phiIndex = 0; phiIndex < sh.anglemapParams.phiWidth; phiIndex++) {
                angleMap.phi_CDFsums[thetaIndex] += angleMap.pdf[thetaIndex*sh.anglemapParams.phiWidth + phiIndex];
            }
            angleMap.theta_CDFsum += angleMap.phi_CDFsums[thetaIndex];
        }
        if (!angleMap.theta_CDFsum) {
            std::stringstream err; err << "Facet " << globalId + 1 << " has all-zero recorded angle map.";
            throw std::runtime_error(err.str().c_str());
            return false;
        }

        //Second pass: write CDFs
        double thetaNormalizingFactor = 1.0 / (double)angleMap.theta_CDFsum;
        for (size_t thetaIndex = 0; thetaIndex < (sh.anglemapParams.thetaLowerRes + sh.anglemapParams.thetaHigherRes); thetaIndex++) {
            if (angleMap.theta_CDFsum == 0) { //no hits in this line, generate CDF of uniform distr.
                angleMap.theta_CDF[thetaIndex] = (0.5 + (double)thetaIndex) / (double)(sh.anglemapParams.thetaLowerRes + sh.anglemapParams.thetaHigherRes);
            }
            else {
                if (thetaIndex == 0) {
                    //First CDF value, covers half of first segment
                    angleMap.theta_CDF[thetaIndex] = 0.5 * (double)angleMap.phi_CDFsums[0] * thetaNormalizingFactor;
                }
                else {
                    //value covering second half of last segment and first of current segment
                    angleMap.theta_CDF[thetaIndex] = angleMap.theta_CDF[thetaIndex - 1] + (double)(angleMap.phi_CDFsums[thetaIndex - 1] + angleMap.phi_CDFsums[thetaIndex])*0.5*thetaNormalizingFactor;
                }
            }
            double phiNormalizingFactor = 1.0 / (double)angleMap.phi_CDFsums[thetaIndex];
            for (size_t phiIndex = 0; phiIndex < sh.anglemapParams.phiWidth; phiIndex++) {
                size_t index = sh.anglemapParams.phiWidth * thetaIndex + phiIndex;
                if (angleMap.phi_CDFsums[thetaIndex] == 0) { //no hits in this line, create CDF of uniform distr.
                    angleMap.phi_CDFs[index] = (0.5 + (double)phiIndex) / (double)sh.anglemapParams.phiWidth;
                }
                else {
                    if (phiIndex == 0) {
                        //First CDF value, covers half of first segment
                        angleMap.phi_CDFs[index] = 0.5 * (double)angleMap.pdf[sh.anglemapParams.phiWidth * thetaIndex] * phiNormalizingFactor;
                    }
                    else {
                        //value covering second half of last segment and first of current segment
                        angleMap.phi_CDFs[index] = angleMap.phi_CDFs[sh.anglemapParams.phiWidth * thetaIndex + phiIndex - 1] + (double)(angleMap.pdf[sh.anglemapParams.phiWidth * thetaIndex + phiIndex - 1] + angleMap.pdf[sh.anglemapParams.phiWidth * thetaIndex + phiIndex])*0.5*phiNormalizingFactor;
                    }
                }
            }
        }
    }
    else {
        //Record mode, create pdf vector
        angleMap.pdf.resize(sh.anglemapParams.GetMapSize());
    }

    if(sh.anglemapParams.record)
        angleMapSize += sh.anglemapParams.GetDataSize();
    return angleMapSize;
}

void SubprocessFacet::InitializeOutgassingMap()
{
    if (sh.useOutgassingFile) {
        //Precalc actual outgassing map width and height for faster generation:
        ogMap.outgassingMapWidthD = sh.U.Norme() * ogMap.outgassingFileRatio;
        ogMap.outgassingMapHeightD = sh.V.Norme() * ogMap.outgassingFileRatio;
        size_t nbE = ogMap.outgassingMapWidth*ogMap.outgassingMapHeight;
        // TODO: Check with molflow_threaded e10c2a6f and 66b89ac7 if right
        // making a copy shouldn't be necessary as i will never get changed before use
        //outgassingMapWindow = facetRef->outgassingMapWindow; //init by copying pdf
        ogMap.outgassingMap_cdf = ogMap.outgassingMap;
        for (size_t i = 1; i < nbE; i++) {
            ogMap.outgassingMap_cdf[i] = ogMap.outgassingMap_cdf[i - 1] + ogMap.outgassingMap_cdf[i]; //Convert p.d.f to cumulative distr.
        }
    }
}

bool SubprocessFacet::InitializeLinkAndVolatile(const size_t & id)
{
    if (sh.superDest || sh.isVolatile) {
        // Link or volatile facet, overides facet settings
        // Must be full opaque and 0 sticking
        // (see SimulationMC.c::PerformBounce)
        //sh.isOpaque = true;
        sh.opacity = 1.0;
        sh.opacity_paramId = -1;
        sh.sticking = 0.0;
        sh.sticking_paramId = -1;
    }
    return true;
}

/*void SubprocessFacet::ResetCounter() {
    std::fill(tmpCounter.begin(), tmpCounter.end(), FacetHitBuffer());
}

void SubprocessFacet::ResizeCounter(size_t nbMoments) {
    tmpCounter = std::vector<FacetHitBuffer>(nbMoments + 1); //Includes 0-init
    tmpCounter.shrink_to_fit();
    tmpHistograms = std::vector<FacetHistogramBuffer>(nbMoments + 1);
    tmpHistograms.shrink_to_fit();
}*/

/**
* \brief Constructor for cereal initialization
*/
SubprocessFacet::SubprocessFacet() : Facet() {
    isReady = false;
    globalId = 0;
}

/**
* \brief Constructor with initialisation based on the number of indices/facets
* \param nbIndex number of indices/facets
*/
SubprocessFacet::SubprocessFacet(size_t nbIndex) : Facet(nbIndex) {
    isReady = false;
    globalId = 0;
    indices.resize(nbIndex);                    // Ref to Geometry Vector3d
    vertices2.resize(nbIndex);
}

/**
* \brief Calculates the hits size for a single facet which is necessary for hits dataport
* \param nbMoments amount of moments
* \return calculated size of the facet hits
*/
size_t SubprocessFacet::GetHitsSize(size_t nbMoments) const { //for hits dataport
    return   (1 + nbMoments)*(
            sizeof(FacetHitBuffer) +
            +(sh.isTextured ? (sh.texWidth*sh.texHeight * sizeof(TextureCell)) : 0)
            + (sh.isProfile ? (PROFILE_SIZE * sizeof(ProfileSlice)) : 0)
            + (sh.countDirection ? (sh.texWidth*sh.texHeight * sizeof(DirectionCell)) : 0)
            + sh.facetHistogramParams.GetDataSize()
    ) + (sh.anglemapParams.record ? (sh.anglemapParams.GetRecordedDataSize()) : 0);

}

size_t SubprocessFacet::GetMemSize() const {
    size_t sum = 0;
    sum += sizeof (SubprocessFacet);
    sum += sizeof (size_t) * indices.capacity();
    sum += sizeof (Vector2d) * vertices2.capacity();
    sum += sizeof (double) * textureCellIncrements.capacity();
    sum += sizeof (bool) * largeEnough.capacity();
    sum += sizeof (double) * ogMap.outgassingMap.capacity();
    sum += angleMap.GetMemSize();
    return sum;
}

/**
* \brief Initialises geometry properties that haven't been loaded from file
* \return error code: 0=no error, 1=error
*/
int SimulationModel::InitialiseFacets() {
    for (auto& facet : facets) {
        // Main facet params
        // Current facet
        //SubprocessFacet *f = model->facets[i];
        CalculateFacetParams(&facet);

        // Set some texture parameters
        // bool Facet::SetTexture(double width, double height, bool useMesh)
        if (facet.sh.texWidthD * facet.sh.texHeightD > 0.0000001) {
            const double ceilCutoff = 0.9999999;
            facet.sh.texWidth = (int) std::ceil(facet.sh.texWidthD *
                                                ceilCutoff); //0.9999999: cut the last few digits (convert rounding error 1.00000001 to 1, not 2)
            facet.sh.texHeight = (int) std::ceil(facet.sh.texHeightD * ceilCutoff);
        } else {
            facet.sh.texWidth = 0;
            facet.sh.texHeight = 0;
            facet.sh.texWidthD = 0.0;
            facet.sh.texHeightD = 0.0;
        }
    }

    return 0;
}
/*!
 * @brief Calculates various facet parameters without sanity checking @see Geometry::CalculateFacetParams(Facet* f)
 * @param f individual subprocess facet
 */
void SimulationModel::CalculateFacetParams(SubprocessFacet* f) {
    // Calculate facet normal
    Vector3d p0 = vertices3[f->indices[0]];
    Vector3d v1;
    Vector3d v2;
    bool consecutive = true;
    size_t ind = 2;

    // TODO: Handle possible collinear consequtive vectors
    size_t i0 = f->indices[0];
    size_t i1 = f->indices[1];
    while (ind < f->sh.nbIndex && consecutive) {
        size_t i2 = f->indices[ind++];

        v1 = vertices3[i1] - vertices3[i0]; // v1 = P0P1
        v2 = vertices3[i2] - vertices3[i1]; // v2 = P1P2
        f->sh.N = CrossProduct(v1, v2);              // Cross product
        consecutive = (f->sh.N.Norme() < 1e-11);
    }
    f->sh.N = f->sh.N.Normalized();                  // Normalize

    // Calculate Axis Aligned Bounding Box
    f->sh.bb.min = Vector3d(1e100, 1e100, 1e100);
    f->sh.bb.max = Vector3d(-1e100, -1e100, -1e100);

    for (const auto& i : f->indices) {
        const Vector3d& p = vertices3[i];
        f->sh.bb.min.x = std::min(f->sh.bb.min.x,p.x);
        f->sh.bb.min.y = std::min(f->sh.bb.min.y, p.y);
        f->sh.bb.min.z = std::min(f->sh.bb.min.z, p.z);
        f->sh.bb.max.x = std::max(f->sh.bb.max.x, p.x);
        f->sh.bb.max.y = std::max(f->sh.bb.max.y, p.y);
        f->sh.bb.max.z = std::max(f->sh.bb.max.z, p.z);
    }

    // Facet center (AxisAlignedBoundingBox center)
    f->sh.center = 0.5 * (f->sh.bb.max + f->sh.bb.min);

    // Plane equation
    //double A = f->sh.N.x;
    //double B = f->sh.N.y;
    //double C = f->sh.N.z;
    //double D = -Dot(f->sh.N, p0);

    Vector3d p1 = vertices3[f->indices[1]];

    Vector3d U, V;

    U = (p1 - p0).Normalized(); //First side

    // Construct a normal vector V:
    V = CrossProduct(f->sh.N, U); // |U|=1 and |N|=1 => |V|=1

    // u,v vertices (we start with p0 at 0,0)
    f->vertices2[0].u = 0.0;
    f->vertices2[0].v = 0.0;
    Vector2d BBmin; BBmin.u = 0.0; BBmin.v = 0.0;
    Vector2d BBmax; BBmax.u = 0.0; BBmax.v = 0.0;

    for (size_t j = 1; j < f->sh.nbIndex; j++) {
        Vector3d p = vertices3[f->indices[j]];
        Vector3d v = p - p0;
        f->vertices2[j].u = Dot(U, v);  // Project p on U along the V direction
        f->vertices2[j].v = Dot(V, v);  // Project p on V along the U direction

        // Bounds
        BBmax.u  = std::max(BBmax.u , f->vertices2[j].u);
        BBmax.v = std::max(BBmax.v, f->vertices2[j].v);
        BBmin.u = std::min(BBmin.u, f->vertices2[j].u);
        BBmin.v = std::min(BBmin.v, f->vertices2[j].v);
    }

    // Calculate facet area (Meister/Gauss formula)
    double area = 0.0;
    for (size_t j = 0; j < f->sh.nbIndex; j++) {
        size_t j_next = Next(j,f->sh.nbIndex);
        area += f->vertices2[j].u*f->vertices2[j_next].v - f->vertices2[j_next].u*f->vertices2[j].v; //Equal to Z-component of vectorial product
    }
    if (area > 0.0) {

    }
    else if (area < 0.0) {
        //This is a case where a concave facet doesn't obey the right-hand rule:
        //it happens when the first rotation (usually around the second index) is the opposite as the general outline rotation

        //Do a flip
        f->sh.N = -1.0 * f->sh.N;
        V = -1.0 * V;
        BBmin.v = BBmax.v = 0.0;
        for (auto& v : f->vertices2) {
            v.v = -1.0 * v.v;
            BBmax.v = std::max(BBmax.v, v.v);
            BBmin.v = std::min(BBmin.v, v.v);
        }
    }

    f->sh.area = std::abs(0.5 * area);

    // Compute the 2D basis (O,U,V)
    double uD = (BBmax.u - BBmin.u);
    double vD = (BBmax.v - BBmin.v);

    // Origin
    f->sh.O = p0 + BBmin.u * U + BBmin.v * V;

    // Rescale U and V vector
    f->sh.nU = U;
    f->sh.U = U * uD;

    f->sh.nV = V;
    f->sh.V = V * vD;

    f->sh.Nuv = CrossProduct(f->sh.U,f->sh.V); //Not normalized normal vector

    // Rescale u,v coordinates
    for (auto& p : f->vertices2) {
        p.u = (p.u - BBmin.u) / uD;
        p.v = (p.v - BBmin.v) / vD;
    }

#if defined(MOLFLOW)
    f->sh.maxSpeed = 4.0 * std::sqrt(2.0*8.31*f->sh.temperature / 0.001 / wp.gasMass);
#endif
}

/**
* \brief Do calculations necessary before launching simulation
* determine latest moment
* Generate integrated desorption functions
* match parameters
* Generate speed distribution functions
* Angle map
*/
void SimulationModel::PrepareToRun() {

    if(sh.nbFacet != facets.size()) {
        std::cerr << "Facet structure not properly initialized, size mismatch: " << sh.nbFacet << " / " << facets.size() << "\n";
        exit(0);
    }
    //determine latest moment
    wp.latestMoment = 1E-10;
    if(!tdParams.moments.empty())
        wp.latestMoment = (tdParams.moments.end()-1)->first + (tdParams.moments.end()-1)->second / 2.0;

    std::set<size_t> desorptionParameterIDs;

    //Check and calculate various facet properties for time dependent simulations (CDF, ID )
    for (size_t i = 0; i < sh.nbFacet; i++) {
        SubprocessFacet& facet = facets[i];
        // TODO: Find a solution to integrate catalog parameters
        if(facet.sh.outgassing_paramId >= (int) tdParams.parameters.size()){
            char tmp[256];
            sprintf(tmp, "Facet #%zd: Outgassing parameter \"%d\" isn't defined.", i + 1, facet.sh.outgassing_paramId);
            throw Error(tmp);
        }
        if(facet.sh.opacity_paramId >= (int) tdParams.parameters.size()){
            char tmp[256];
            sprintf(tmp, "Facet #%zd: Opacity parameter \"%d\" isn't defined.", i + 1, facet.sh.opacity_paramId);
            throw Error(tmp);
        }
        if(facet.sh.sticking_paramId >= (int) tdParams.parameters.size()){
            char tmp[256];
            sprintf(tmp, "Facet #%zd: Sticking parameter \"%d\" isn't defined.", i + 1, facet.sh.sticking_paramId);
            throw Error(tmp);
        }

        if (facet.sh.outgassing_paramId >= 0) { //if time-dependent desorption
            int id = IDGeneration::GetIDId(desorptionParameterIDs, facet.sh.outgassing_paramId);
            if (id >= 0)
                facet.sh.IDid = id; //we've already generated an ID for this temperature
            else {
                auto[id_new, id_vec] = IDGeneration::GenerateNewID(desorptionParameterIDs, facet.sh.outgassing_paramId, this);
                facet.sh.IDid = id_new;
                tdParams.IDs.emplace_back(std::move(id_vec));
            }
        }

        // Generate speed distribution functions
        std::set<double> temperatureList;
        int id = CDFGeneration::GetCDFId(temperatureList, facet.sh.temperature);
        if (id >= 0)
            facet.sh.CDFid = id; //we've already generated a CDF for this temperature
        else {
            auto[id, cdf_vec] = CDFGeneration::GenerateNewCDF(temperatureList, facet.sh.temperature, wp.gasMass);
            facet.sh.CDFid = id;
            tdParams.CDFs.emplace_back(cdf_vec);
        }
        //Angle map
        if (facet.sh.desorbType == DES_ANGLEMAP) {
            if (!facet.sh.anglemapParams.hasRecorded) {
                char tmp[256];
                sprintf(tmp, "Facet #%zd: Uses angle map desorption but doesn't have a recorded angle map.", i + 1);
                throw Error(tmp);
            }
            if (facet.sh.anglemapParams.record) {
                char tmp[256];
                sprintf(tmp, "Facet #%zd: Can't RECORD and USE angle map desorption at the same time.", i + 1);
                throw Error(tmp);
            }
        }
    }

    CalcTotalOutgassing();

    initialized = true;
}

/**
* \brief Compute the outgassing of all source facet depending on the mode (file, regular, time-dependent) and set it to the global settings
*/
void SimulationModel::CalcTotalOutgassing() {
    // Compute the outgassing of all source facet
    double totalDesorbedMolecules = 0.0;
    double finalOutgassingRate_Pa_m3_sec = 0.0;
    double finalOutgassingRate = 0.0;

    const double latestMoment = wp.latestMoment;

    for (size_t i = 0; i < sh.nbFacet; i++) {
        SubprocessFacet& facet = facets[i];
        if (facet.sh.desorbType != DES_NONE) { //there is a kind of desorption
            if (facet.sh.useOutgassingFile) { //outgassing file
                auto& ogMap = facet.ogMap;
                for (size_t l = 0; l < (ogMap.outgassingMapWidth * ogMap.outgassingMapHeight); l++) {
                    totalDesorbedMolecules += latestMoment * ogMap.outgassingMap[l] / (1.38E-23 * facet.sh.temperature);
                    finalOutgassingRate += ogMap.outgassingMap[l] / (1.38E-23 * facet.sh.temperature);
                    finalOutgassingRate_Pa_m3_sec += ogMap.outgassingMap[l];
                }
            } else { //regular outgassing
                if (facet.sh.outgassing_paramId == -1) { //constant outgassing
                    totalDesorbedMolecules += latestMoment * facet.sh.outgassing / (1.38E-23 * facet.sh.temperature);
                    finalOutgassingRate +=
                            facet.sh.outgassing / (1.38E-23 * facet.sh.temperature);  //Outgassing molecules/sec
                    finalOutgassingRate_Pa_m3_sec += facet.sh.outgassing;
                } else { //time-dependent outgassing
                    totalDesorbedMolecules += tdParams.IDs[facet.sh.IDid].back().second / (1.38E-23 * facet.sh.temperature);
                    size_t lastIndex = tdParams.parameters[facet.sh.outgassing_paramId].GetSize() - 1;
                    double finalRate_mbar_l_s = tdParams.parameters[facet.sh.outgassing_paramId].GetY(lastIndex);
                    finalOutgassingRate +=
                            finalRate_mbar_l_s * 0.100 / (1.38E-23 * facet.sh.temperature); //0.1: mbar*l/s->Pa*m3/s
                    finalOutgassingRate_Pa_m3_sec += finalRate_mbar_l_s * 0.100;
                }
            }
        }
    }

    wp.totalDesorbedMolecules = totalDesorbedMolecules;
    wp.finalOutgassingRate_Pa_m3_sec = finalOutgassingRate_Pa_m3_sec;
    wp.finalOutgassingRate = finalOutgassingRate;
}

SimulationModel::~SimulationModel() {

}

/**
* \brief Assign operator
* \param src reference to source object
* \return address of this
*/
GlobalSimuState& GlobalSimuState::operator=(const GlobalSimuState & src) {
    //Copy all but mutex
    facetStates = src.facetStates;
    globalHistograms = src.globalHistograms;
    globalHits = src.globalHits;
    initialized = src.initialized;
    return *this;
}

/**
* \brief Assign operator
* \param src reference to source object
* \return address of this
*/
GlobalSimuState& GlobalSimuState::operator+=(const GlobalSimuState & src) {
    //Copy all but mutex
    facetStates += src.facetStates;
    globalHistograms += src.globalHistograms;
    globalHits += src.globalHits;
    return *this;
}

/**
* \brief Clears simulation state
*/
void GlobalSimuState::clear() {
    tMutex.lock();
    globalHits = GlobalHitBuffer();
    globalHistograms.clear();
    facetStates.clear();
    initialized = false;
    tMutex.unlock();
}

/**
* \brief Constructs the 'dpHit' structure to hold all results, zero-init
* \param w Worker handle
*/
void GlobalSimuState::Resize(const SimulationModel &model) { //Constructs the 'dpHit' structure to hold all results, zero-init
    //LockMutex(mutex);
    tMutex.lock();
    size_t nbF = model.sh.nbFacet;
    size_t nbMoments = model.tdParams.moments.size();
    std::vector<FacetState>(nbF).swap(facetStates);

    if(!model.facets.empty()) {
        for (size_t i = 0; i < nbF; i++) {
            auto &sFac = model.facets[i];
            if (sFac.globalId != i) {
                std::cerr << "Facet ID mismatch! : " << sFac.globalId << " / " << i << "\n";
                exit(0);
            }
            FacetMomentSnapshot facetMomentTemplate;
            facetMomentTemplate.histogram.Resize(sFac.sh.facetHistogramParams);
            facetMomentTemplate.direction = std::vector<DirectionCell>(
                    sFac.sh.countDirection ? sFac.sh.texWidth * sFac.sh.texHeight : 0);
            facetMomentTemplate.profile = std::vector<ProfileSlice>(sFac.sh.isProfile ? PROFILE_SIZE : 0);
            facetMomentTemplate.texture = std::vector<TextureCell>(
                    sFac.sh.isTextured ? sFac.sh.texWidth * sFac.sh.texHeight : 0);
            //No init for hits
            facetStates[i].momentResults = std::vector<FacetMomentSnapshot>(1 + nbMoments, facetMomentTemplate);
            if (sFac.sh.anglemapParams.record)
                facetStates[i].recordedAngleMapPdf = std::vector<size_t>(sFac.sh.anglemapParams.GetMapSize());
        }
    }
    /*for (size_t i = 0; i < nbF; i++) {
        const SubprocessFacet& sFac = facets[i];
        FacetMomentSnapshot facetMomentTemplate;
        facetMomentTemplate.histogram.Resize(sFac.sh.facetHistogramParams);
        facetMomentTemplate.direction = std::vector<DirectionCell>(sFac.sh.countDirection ? sFac.sh.texWidth*sFac.sh.texHeight : 0);
        facetMomentTemplate.profile = std::vector<ProfileSlice>(sFac.sh.isProfile ? PROFILE_SIZE : 0);
        facetMomentTemplate.texture = std::vector<TextureCell>(sFac.sh.isTextured ? sFac.sh.texWidth*sFac.sh.texHeight : 0);
        //No init for hits
        facetStates[i].momentResults = std::vector<FacetMomentSnapshot>(1 + nbMoments, facetMomentTemplate);
        if (sFac.sh.anglemapParams.record) facetStates[i].recordedAngleMapPdf = std::vector<size_t>(sFac.sh.anglemapParams.GetMapSize());
    }*/
    //Global histogram

    FacetHistogramBuffer globalHistTemplate; globalHistTemplate.Resize(model.wp.globalHistogramParams);
    globalHistograms = std::vector<FacetHistogramBuffer>(1 + nbMoments, globalHistTemplate);
    initialized = true;
    //ReleaseMutex(mutex);
    tMutex.unlock();
}

/**
* \brief zero-init for all structures
*/
void GlobalSimuState::Reset() {
    //LockMutex(mutex);
    tMutex.lock();
    for (auto& h : globalHistograms) {
        ZEROVECTOR(h.distanceHistogram);
        ZEROVECTOR(h.nbHitsHistogram);
        ZEROVECTOR(h.timeHistogram);
    }
    memset(&globalHits, 0, sizeof(globalHits)); //Plain old data
    for (auto& state : facetStates) {
        ZEROVECTOR(state.recordedAngleMapPdf);
        for (auto& m : state.momentResults) {
            ZEROVECTOR(m.histogram.distanceHistogram);
            ZEROVECTOR(m.histogram.nbHitsHistogram);
            ZEROVECTOR(m.histogram.timeHistogram);
            std::vector<DirectionCell>(m.direction.size()).swap(m.direction);
            std::vector<TextureCell>(m.texture.size()).swap(m.texture);
            std::vector<ProfileSlice>(m.profile.size()).swap(m.profile);
            memset(&(m.hits), 0, sizeof(m.hits));
        }
    }
    tMutex.unlock();
    //ReleaseMutex(mutex);
}

std::pair<int, int>
GlobalSimuState::Compare(const GlobalSimuState &lhsGlobHit, const GlobalSimuState &rhsGlobHit, double cmpThreshold) {

    //std::ofstream cmpFile("cmpFile.txt");
    size_t globalErrNb = 0;
    size_t facetErrNb = 0;

    std::stringstream cmpFile;

    // Sanity check
    {
        if(lhsGlobHit.globalHits.globalHits.nbDesorbed == 0 && rhsGlobHit.globalHits.globalHits.nbDesorbed == 0){
            cmpFile << "[Global][desorp] Neither state has recorded desorptions\n";
            ++globalErrNb;
        }
        else if (lhsGlobHit.globalHits.globalHits.nbDesorbed == 0){
            cmpFile << "[Global][desorp] First state has no recorded desorptions\n";
            ++globalErrNb;
        }
        else if (rhsGlobHit.globalHits.globalHits.nbDesorbed == 0){
            cmpFile << "[Global][desorp] Second state has no recorded desorptions\n";
            ++globalErrNb;
        }

        if(globalErrNb){
            std::cout << cmpFile.str() << "\n";
            return std::make_pair(globalErrNb, std::numeric_limits<int>::max());
        }
    }

    {
        double absRatio = lhsGlobHit.globalHits.globalHits.nbAbsEquiv / lhsGlobHit.globalHits.globalHits.nbDesorbed;
        double absRatio_rhs = rhsGlobHit.globalHits.globalHits.nbAbsEquiv / rhsGlobHit.globalHits.globalHits.nbDesorbed;
        if (!IsEqual(absRatio, absRatio_rhs, cmpThreshold)) {
            cmpFile << "[Global][absRatio] has large difference: "<<std::abs(absRatio - absRatio_rhs)<<"\n";
            ++globalErrNb;
        }
    }

    {
        double hitRatio = (double)lhsGlobHit.globalHits.globalHits.nbMCHit / lhsGlobHit.globalHits.globalHits.nbDesorbed;
        double hitRatio_rhs = (double)rhsGlobHit.globalHits.globalHits.nbMCHit / rhsGlobHit.globalHits.globalHits.nbDesorbed;
        if (!IsEqual(hitRatio, hitRatio_rhs, cmpThreshold)) {
            cmpFile << "[Global][hitRatio] has large difference: "<<
            std::abs(hitRatio - hitRatio_rhs)<<"\n";
            cmpFile << lhsGlobHit.globalHits.globalHits.nbMCHit <<
            " / "<< lhsGlobHit.globalHits.globalHits.nbDesorbed <<
             " vs "<< rhsGlobHit.globalHits.globalHits.nbMCHit <<
              " / "<< rhsGlobHit.globalHits.globalHits.nbDesorbed <<"\n";
            ++globalErrNb;
        }
    }

    if (!IsEqual(lhsGlobHit.globalHits.globalHits.sum_v_ort, rhsGlobHit.globalHits.globalHits.sum_v_ort, cmpThreshold)) {
        cmpFile << "[Global][sum_v_ort] has large difference: "<<std::abs(lhsGlobHit.globalHits.globalHits.sum_v_ort - rhsGlobHit.globalHits.globalHits.sum_v_ort)<<"\n";
        ++globalErrNb;
    }
    if (!IsEqual(lhsGlobHit.globalHits.globalHits.sum_1_per_velocity, rhsGlobHit.globalHits.globalHits.sum_1_per_velocity, cmpThreshold)) {
        cmpFile << "[Global][sum_1_per_velocity] has large difference: "<<std::abs(lhsGlobHit.globalHits.globalHits.sum_1_per_velocity - rhsGlobHit.globalHits.globalHits.sum_1_per_velocity)<<"\n";
        ++globalErrNb;
    }
    if (!IsEqual(lhsGlobHit.globalHits.globalHits.sum_1_per_ort_velocity, rhsGlobHit.globalHits.globalHits.sum_1_per_ort_velocity, cmpThreshold)) {
        cmpFile << "[Global][sum_1_per_ort_velocity] has large difference: "<<std::abs(lhsGlobHit.globalHits.globalHits.sum_1_per_ort_velocity - rhsGlobHit.globalHits.globalHits.sum_1_per_ort_velocity)<<"\n";
        ++globalErrNb;
    }


    // Histogram
    /*for (size_t hIndex = 0; hIndex < saveHistSize; ++hIndex) {
        if(!IsEqual(nbHitsHistogram[hIndex],nbHitsHistogram_rhs[hIndex], cmpThreshold)){
            cmpFile << "[Global][Hist][Bounces][Ind="<<hIndex<<"] has large difference: "<<std::abs(nbHitsHistogram[hIndex] - nbHitsHistogram_rhs[hIndex])<<"\n";
        }
    }
    for (size_t hIndex = 0; hIndex < saveHistSize; ++hIndex) {
        if(!IsEqual(distanceHistogram[hIndex],distanceHistogram_rhs[hIndex], cmpThreshold)){
            cmpFile << "[Global][Hist][Dist][Ind="<<hIndex<<"] has large difference: "<<std::abs(distanceHistogram[hIndex] - distanceHistogram_rhs[hIndex])<<"\n";
        }
    }
    for (size_t hIndex = 0; hIndex < saveHistSize; ++hIndex) {
        if(!IsEqual(timeHistogram[hIndex],timeHistogram_rhs[hIndex], cmpThreshold)){
            cmpFile << "[Global][Hist][Time][Ind="<<hIndex<<"] has large difference: "<<std::abs(timeHistogram[hIndex] - timeHistogram_rhs[hIndex])<<"\n";
        }
    }*/

    // Lower facet threshold as less hits
    cmpThreshold *= 10.0;

    // facets
    for(int facetId = 0; facetId < lhsGlobHit.facetStates.size(); ++facetId)
    {//cmp
        auto& facetCounter_lhs = lhsGlobHit.facetStates[facetId].momentResults[0];
        auto& facetCounter_rhs = rhsGlobHit.facetStates[facetId].momentResults[0];

        // If one facet doesn't have any hits recorded, comparison is pointless, so just skip to next facet
        if (facetCounter_lhs.hits.nbMCHit == 0 && facetCounter_rhs.hits.nbMCHit == 0){
            //cmpFile << "[Facet]["<<facetId<<"][hits] Neither state has recorded hits for this facet\n";
            continue;
        }
        else if (facetCounter_lhs.hits.nbMCHit == 0 && facetCounter_rhs.hits.nbMCHit > 0){
            cmpFile << "[Facet]["<<facetId<<"][hits] First state has no recorded hits for this facet\n";
            ++facetErrNb;
            continue;
        }
        else if (facetCounter_lhs.hits.nbMCHit > 0 && facetCounter_rhs.hits.nbMCHit == 0){
            cmpFile << "[Facet]["<<facetId<<"][hits] Second state has no recorded hits for this facet\n";
            ++facetErrNb;
            continue;
        }

        double scale = 1.0 / lhsGlobHit.globalHits.globalHits.nbDesorbed; // getmolpertp
        double scale_rhs = 1.0 / rhsGlobHit.globalHits.globalHits.nbDesorbed;
        double fullScale = 1.0;
        if (facetCounter_lhs.hits.nbMCHit > 0 || facetCounter_lhs.hits.nbDesorbed > 0) {
            if (facetCounter_lhs.hits.nbAbsEquiv > 0.0 || facetCounter_lhs.hits.nbDesorbed > 0) {//otherwise save calculation time
                fullScale = 1.0 - (facetCounter_lhs.hits.nbAbsEquiv + (double)facetCounter_lhs.hits.nbDesorbed) / (facetCounter_lhs.hits.nbHitEquiv + (double)facetCounter_lhs.hits.nbDesorbed) / 2.0;
            }
        }

        double fullScale_rhs = 1.0;
        if (facetCounter_rhs.hits.nbMCHit > 0 || facetCounter_rhs.hits.nbDesorbed > 0) {
            if (facetCounter_rhs.hits.nbAbsEquiv > 0.0 || facetCounter_rhs.hits.nbDesorbed > 0) {//otherwise save calculation time
                fullScale_rhs = 1.0 - (facetCounter_rhs.hits.nbAbsEquiv + (double)facetCounter_rhs.hits.nbDesorbed) / (facetCounter_rhs.hits.nbHitEquiv + (double)facetCounter_rhs.hits.nbDesorbed) / 2.0;
            }
        }

        fullScale *= scale;
        fullScale_rhs *= scale_rhs;

        double sumHitDes = facetCounter_lhs.hits.nbHitEquiv + static_cast<double>(facetCounter_lhs.hits.nbDesorbed);
        double sumHitDes_rhs = facetCounter_rhs.hits.nbHitEquiv + static_cast<double>(facetCounter_rhs.hits.nbDesorbed);

        {
            double hitRatio = facetCounter_lhs.hits.nbHitEquiv * scale;
            double hitRatio_rhs = facetCounter_rhs.hits.nbHitEquiv * scale_rhs;
            if (!IsEqual(hitRatio, hitRatio_rhs, cmpThreshold)) {
                cmpFile << "[Facet]["<<facetId<<"][hitRatio] has large difference: "<<std::abs(hitRatio - hitRatio_rhs)<<"\n";
                ++facetErrNb;
            }
        }

        {
            double absRatio = facetCounter_lhs.hits.nbAbsEquiv / facetCounter_lhs.hits.nbMCHit;
            double absRatio_rhs = facetCounter_rhs.hits.nbAbsEquiv / facetCounter_rhs.hits.nbMCHit;
            if (!IsEqual(absRatio, absRatio_rhs, cmpThreshold)) {
                cmpFile << "[Facet]["<<facetId<<"][absRatio] has large difference: "<<std::abs(absRatio - absRatio_rhs)<<"\n";
                ++facetErrNb;
            }
        }

        {
            double desRatio = (double)facetCounter_lhs.hits.nbDesorbed / facetCounter_lhs.hits.nbMCHit;
            double desRatio_rhs = (double)facetCounter_rhs.hits.nbDesorbed / facetCounter_rhs.hits.nbMCHit;
            if (!IsEqual(desRatio, desRatio_rhs, cmpThreshold)) {
                cmpFile << "[Facet]["<<facetId<<"][desRatio] has large difference: "<<std::abs(desRatio - desRatio_rhs)<<"\n";
                ++facetErrNb;
            }
        }

        if (!IsEqual(facetCounter_lhs.hits.sum_v_ort * scale, facetCounter_rhs.hits.sum_v_ort * scale_rhs, cmpThreshold)) {
            cmpFile << "[Facet]["<<facetId<<"][sum_v_ort] has large difference: "<<std::abs(facetCounter_lhs.hits.sum_v_ort * scale - facetCounter_rhs.hits.sum_v_ort * scale_rhs)<<"\n";
        }
        if (!IsEqual(sumHitDes / facetCounter_lhs.hits.sum_1_per_velocity, sumHitDes_rhs / facetCounter_rhs.hits.sum_1_per_velocity, cmpThreshold)) {
            cmpFile << "[Facet]["<<facetId<<"][sum_1_per_velocity] has large difference: "<<std::abs(sumHitDes / facetCounter_lhs.hits.sum_1_per_velocity - sumHitDes_rhs / facetCounter_rhs.hits.sum_1_per_velocity)<< " ===> " << std::abs(sumHitDes / facetCounter_lhs.hits.sum_1_per_velocity - sumHitDes_rhs / facetCounter_rhs.hits.sum_1_per_velocity)/(sumHitDes / facetCounter_lhs.hits.sum_1_per_velocity) <<"\n";
        }
        if (!IsEqual(facetCounter_lhs.hits.sum_1_per_ort_velocity * fullScale, facetCounter_rhs.hits.sum_1_per_ort_velocity* fullScale_rhs, cmpThreshold)) {
            cmpFile << "[Facet]["<<facetId<<"][sum_1_per_ort_velocity] has large difference: "<<std::abs(facetCounter_lhs.hits.sum_1_per_ort_velocity * fullScale - facetCounter_rhs.hits.sum_1_per_ort_velocity * fullScale_rhs)<< " ===> " << std::abs(facetCounter_lhs.hits.sum_1_per_ort_velocity * fullScale - facetCounter_rhs.hits.sum_1_per_ort_velocity * fullScale_rhs)/(facetCounter_lhs.hits.sum_1_per_ort_velocity * fullScale) << "\n";
        }
    }

    /*//profile
    {
        for (id = 0; id < profile.size(); ++id) {
            if(!IsEqual(profile[id].countEquiv / sumHitDes,profile_rhs[id].countEquiv / sumHitDes_rhs, cmpThreshold)){
                cmpFile << "[Facet]["<<facetId<<"][Profile][Ind="<<id<<"][countEquiv] has large difference: "<<std::abs(profile[id].countEquiv / sumHitDes - profile_rhs[id].countEquiv / sumHitDes_rhs)/(profile[id].countEquiv / sumHitDes)<< " : " << std::abs(profile[id].countEquiv / sumHitDes) << " - " << (profile_rhs[id].countEquiv / sumHitDes_rhs) << "\n";
            }
        }
        for (id = 0; id < profile.size(); ++id) {
            if(!IsEqual(profile[id].sum_1_per_ort_velocity * scale,profile_rhs[id].sum_1_per_ort_velocity * scale_rhs, cmpThreshold)){
                cmpFile << "[Facet]["<<facetId<<"][Profile][Ind="<<id<<"][sum_1_per_ort_velocity] has large rel difference: "<<std::abs(profile[id].sum_1_per_ort_velocity * scale - profile_rhs[id].sum_1_per_ort_velocity * scale_rhs) / (profile[id].sum_1_per_ort_velocity * scale)<< " : " << std::abs(profile[id].sum_1_per_ort_velocity * scale) << " - " << (profile_rhs[id].sum_1_per_ort_velocity * scale_rhs) <<"\n";
            }
        }
        for (id = 0; id < profile.size(); ++id) { // export_profiles PROFILE_U/V
            if(!IsEqual(profile[id].sum_v_ort * scale,profile_rhs[id].sum_v_ort * scale_rhs, cmpThreshold)){
                cmpFile << "[Facet]["<<facetId<<"][Profile][Ind="<<id<<"][sum_v_ort] has large difference: "<<std::abs(profile[id].sum_v_ort * scale - profile_rhs[id].sum_v_ort * scale_rhs) / (profile[id].sum_v_ort * scale)<< " : " << std::abs(profile[id].sum_v_ort * scale) << " - " << (profile_rhs[id].sum_v_ort * scale_rhs) <<"\n";
            }
        }
    }

    //texture
    {
        for (iy = 0; iy < texHeight_file; iy++) { //MIN: If stored texture is larger, don't read extra cells
            for (ix = 0; ix < texWidth_file; ix++) { //MIN: If stored texture is larger, don't read extra cells
                if(!IsEqual(texture[iy * texWidth_file + ix].countEquiv / sumHitDes ,texture_rhs[iy * texWidth_file + ix].countEquiv/ sumHitDes_rhs, cmpThreshold)){
                    cmpFile << "[Facet]["<<facetId<<"][Texture]["<<ix<<","<<iy<<"][countEquiv] has large rel difference: "<<std::abs(texture[iy * texWidth_file + ix].countEquiv/ sumHitDes - texture_rhs[iy * texWidth_file + ix].countEquiv/ sumHitDes_rhs) / (texture[iy * texWidth_file + ix].countEquiv / sumHitDes)<< " : " << std::abs(texture[iy * texWidth_file + ix].countEquiv / sumHitDes) << " - " << (texture_rhs[iy * texWidth_file + ix].countEquiv / sumHitDes_rhs) <<"\n";
                }
                if(!IsEqual(texture[iy * texWidth_file + ix].sum_1_per_ort_velocity * fullScale,texture_rhs[iy * texWidth_file + ix].sum_1_per_ort_velocity * fullScale_rhs, cmpThreshold)){
                    cmpFile << "[Facet]["<<facetId<<"][Texture]["<<ix<<","<<iy<<"][sum_1_per_ort_velocity] has large rel difference: "<<std::abs(texture[iy * texWidth_file + ix].sum_1_per_ort_velocity  * fullScale - texture_rhs[iy * texWidth_file + ix].sum_1_per_ort_velocity * fullScale_rhs) / (texture[iy * texWidth_file + ix].sum_1_per_ort_velocity  * fullScale)<< " : " << std::abs(texture[iy * texWidth_file + ix].sum_1_per_ort_velocity * fullScale) << " - " << (texture_rhs[iy * texWidth_file + ix].sum_1_per_ort_velocity * fullScale_rhs) <<"\n";
                }
                if(!IsEqual(texture[iy * texWidth_file + ix].sum_v_ort_per_area * scale,texture_rhs[iy * texWidth_file + ix].sum_v_ort_per_area * scale_rhs, cmpThreshold)){
                    cmpFile << "[Facet]["<<facetId<<"][Texture]["<<ix<<","<<iy<<"][sum_v_ort_per_area] has large rel difference: "<<std::abs(texture[iy * texWidth_file + ix].sum_v_ort_per_area  * scale - texture_rhs[iy * texWidth_file + ix].sum_v_ort_per_area * scale_rhs) / (texture[iy * texWidth_file + ix].sum_v_ort_per_area  * scale)<< " : " << std::abs(texture[iy * texWidth_file + ix].sum_v_ort_per_area * scale) << " - " << (texture_rhs[iy * texWidth_file + ix].sum_v_ort_per_area * scale_rhs) <<"\n";
                }
            }
        } // end for comp texture
    }

    //Directions
    {
        for (int iy = 0; iy < dirHeight_file; iy++) {
            for (int ix = 0; ix < dirWidth_file; ix++) {
                if(!IsEqual(dirs[iy * dirWidth_file + ix].count,dirs[iy * dirWidth_file + ix].count, cmpThreshold)){
                    cmpFile << "[Facet]["<<facetId<<"][dirs]["<<ix<<","<<iy<<"][count] has large difference: "<<std::abs((int)dirs[iy * dirWidth_file + ix].count - (int)dirs_rhs[iy * dirWidth_file + ix].count)<<"\n";
                }
                if(!IsEqual(dirs[iy * dirWidth_file + ix].dir.x,dirs[iy * dirWidth_file + ix].dir.x, cmpThreshold)){
                    cmpFile << "[Facet]["<<facetId<<"][dirs]["<<ix<<","<<iy<<"][dir.x] has large difference: "<<std::abs(dirs[iy * dirWidth_file + ix].dir.x - dirs_rhs[iy * dirWidth_file + ix].dir.x)<<"\n";
                }
                if(!IsEqual(dirs[iy * dirWidth_file + ix].dir.y,dirs[iy * dirWidth_file + ix].dir.y, cmpThreshold)){
                    cmpFile << "[Facet]["<<facetId<<"][dirs]["<<ix<<","<<iy<<"][dir.y] has large difference: "<<std::abs(dirs[iy * dirWidth_file + ix].dir.y - dirs_rhs[iy * dirWidth_file + ix].dir.y)<<"\n";
                }
                if(!IsEqual(dirs[iy * dirWidth_file + ix].dir.z,dirs[iy * dirWidth_file + ix].dir.z, cmpThreshold)){
                    cmpFile << "[Facet]["<<facetId<<"][dirs]["<<ix<<","<<iy<<"][dir.z] has large difference: "<<std::abs(dirs[iy * dirWidth_file + ix].dir.z - dirs_rhs[iy * dirWidth_file + ix].dir.z)<<"\n";
                }
            }
        } // end for comp dir
    }

    //facet hist
    {
        for (size_t hIndex = 0; hIndex < saveHistSize; ++hIndex) {
            if(!IsEqual(nbHitsHistogram[hIndex],nbHitsHistogram_rhs[hIndex], cmpThreshold)){
                cmpFile << "[Facet]["<<facetId<<"][Hist][Bounces][Ind="<<hIndex<<"] has large difference: "<<std::abs(nbHitsHistogram[hIndex] - nbHitsHistogram_rhs[hIndex])<<"\n";
            }
        }

        for (size_t hIndex = 0; hIndex < saveHistSize; ++hIndex) {
            if(!IsEqual(distanceHistogram[hIndex],distanceHistogram_rhs[hIndex], cmpThreshold)){
                cmpFile << "[Facet]["<<facetId<<"][Hist][Dist][Ind="<<hIndex<<"] has large difference: "<<std::abs(distanceHistogram[hIndex] - distanceHistogram_rhs[hIndex])<<"\n";
            }
        }

        for (size_t hIndex = 0; hIndex < saveHistSize; ++hIndex) {
            if(!IsEqual(timeHistogram[hIndex],timeHistogram_rhs[hIndex], cmpThreshold)){
                cmpFile << "[Facet]["<<facetId<<"][Hist][Time][Ind="<<hIndex<<"] has large difference: "<<std::abs(timeHistogram[hIndex] - timeHistogram_rhs[hIndex])<<"\n";
            }
        }
    }*/
    std::string cmp_string;
    for(int i = 0; i < 100 && std::getline(cmpFile,cmp_string,'\n'); ++i) {
        std::cout << cmp_string << "\n";
    }
    return std::make_pair(globalErrNb, facetErrNb);
}

/**
* \brief Resize histograms according to sizes in params
* \param params contains data about sizes
*/
/*void FacetHistogramBuffer::Resize(const HistogramParams& params) {
    nbHitsHistogram = std::vector<double>(params.recordBounce ? params.GetBounceHistogramSize() : 0);
    distanceHistogram = std::vector<double>(params.recordDistance ? params.GetDistanceHistogramSize() : 0);
    timeHistogram = std::vector<double>(params.recordTime ? params.GetTimeHistogramSize() : 0);
}*/

/**
* \brief += operator, with simple += of underlying structures
* \param rhs reference object on the right hand
* \return address of this (lhs)
*/
FacetHistogramBuffer& FacetHistogramBuffer::operator+=(const FacetHistogramBuffer & rhs) {
    // if (model.wp.globalHistogramParams.recordBounce)
    this->nbHitsHistogram += rhs.nbHitsHistogram;
    this->distanceHistogram += rhs.distanceHistogram;
    this->timeHistogram += rhs.timeHistogram;
    return *this;
}

/**
* \brief += operator, with simple += of underlying structures
* \param rhs reference object on the right hand
* \return address of this (lhs)
*/
FacetMomentSnapshot& FacetMomentSnapshot::operator+=(const FacetMomentSnapshot & rhs) {
    this->hits += rhs.hits;
    this->profile += rhs.profile;
    this->texture += rhs.texture;
    this->direction += rhs.direction;
    this->histogram += rhs.histogram;
    return *this;
}

/**
* \brief + operator, simply calls implemented +=
* \param rhs reference object on the right hand
* \return address of this (lhs)
*/
FacetMomentSnapshot& FacetMomentSnapshot::operator+(const FacetMomentSnapshot & rhs) {
    *this += rhs;
    return *this;
}

/**
* \brief += operator, with simple += of underlying structures
* \param rhs reference object on the right hand
* \return address of this (lhs)
*/
FacetState& FacetState::operator+=(const FacetState & rhs) {
    // Check in case simulation pdf is empty (record==false) but global pdf is not (hasRecorded==true)
    if(this->recordedAngleMapPdf.size() == rhs.recordedAngleMapPdf.size())
        this->recordedAngleMapPdf += rhs.recordedAngleMapPdf;
    this->momentResults += rhs.momentResults;
    return *this;
}