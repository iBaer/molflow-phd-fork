//
// Created by pascal on 2/5/21.
//

#include <Helper/MathTools.h>
#include <cmath>
#include "AnglemapGeneration.h"

std::tuple<double, int, double>
AnglemapGeneration::GenerateThetaFromAngleMap(const AnglemapParams &anglemapParams, const Anglemap &anglemap,
                                              const double lookupValue) {
    //double lookupValue = randomGenerator.rnd();
    int thetaLowerIndex = my_lower_bound(lookupValue,
                                         anglemap.theta_CDF); //returns line number AFTER WHICH LINE lookup value resides in ( -1 .. size-2 )
    double theta, thetaOvershoot;

    if (thetaLowerIndex == -1) { //first half section
        thetaOvershoot = 0.5 + 0.5 * lookupValue / anglemap.theta_CDF[0]; //between 0.5 and 1
        theta = GetTheta((double) thetaLowerIndex + 0.5 + thetaOvershoot,
                         anglemapParams); //between 0 and the first section end
        return {theta, thetaLowerIndex, thetaOvershoot};
    } else if (thetaLowerIndex == (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes -
                                   1)) { //last half section //can this happen?
        thetaOvershoot = 0.5 * (lookupValue - anglemap.theta_CDF[thetaLowerIndex])
                         / (1.0 - anglemap.theta_CDF[thetaLowerIndex]); //between 0 and 0.5
        theta = GetTheta((double) thetaLowerIndex + 0.5 + thetaOvershoot,
                         anglemapParams); //between 0 and the first section end
        return {theta, thetaLowerIndex, thetaOvershoot};
    } else { //regular section
        if (/*true || */anglemap.phi_CDFsums[thetaLowerIndex] == anglemap.phi_CDFsums[thetaLowerIndex + 1]) {
            //The pdf's slope is 0, linear interpolation
            thetaOvershoot = (lookupValue - anglemap.theta_CDF[thetaLowerIndex]) /
                             (anglemap.theta_CDF[thetaLowerIndex + 1] - anglemap.theta_CDF[thetaLowerIndex]);
            theta = GetTheta((double) thetaLowerIndex + 0.5 + thetaOvershoot, anglemapParams);
        } else {
            //2nd degree interpolation
            // y(x) = ax^2 + bx + c
            // c: CDF value at lower index
            // b: pdf value at lower index
            // a: pdf slope at lower index / 2
            // dy := y - c
            // dx := x - [x at lower index]
            // dy = ax^2 + bx
            // dx = ( -b + sqrt(b^2 +4*a*dy) ) / (2a)
            double thetaStep = GetTheta((double) thetaLowerIndex + 1.5, anglemapParams) -
                               GetTheta((double) thetaLowerIndex + 0.5, anglemapParams);
            double c = anglemap.theta_CDF[thetaLowerIndex]; //CDF value at lower index
            double b = (double) anglemap.phi_CDFsums[thetaLowerIndex] / (double) anglemap.theta_CDFsum /
                       thetaStep; //pdf value at lower index
            double a = 0.5 * ((double) (anglemap.phi_CDFsums[thetaLowerIndex + 1]) -
                              (double) anglemap.phi_CDFsums[thetaLowerIndex]) /
                       (double) anglemap.theta_CDFsum / Sqr(thetaStep); //pdf slope at lower index
            double dy = lookupValue - c;

            double dx = (-b + sqrt(Sqr(b) + 4 * a * dy)) /
                        (2 * a); //Since b>=0 it's the + branch of the +- that is valid for us

            thetaOvershoot = dx / thetaStep;
            theta = GetTheta((double) thetaLowerIndex + 0.5 + thetaOvershoot, anglemapParams);
        }
    }
    return {theta, thetaLowerIndex, thetaOvershoot};
}

/**
* \brief Generates phi angle (azimuth) from angle map
* \param thetaLowerIndex lower index of theta angle of bin in CDF (cummulative distribution function)
* \param thetaOvershoot corresponding to a weight of the previous and next lines
* \param anglemapParams parameters of the angle map
* \param randomGenerator reference to the random number generator (Mersenne Twister)
* \return phi angle
*/

double AnglemapGeneration::GeneratePhiFromAngleMap(const int &thetaLowerIndex, const double &thetaOvershoot,
                                                   const AnglemapParams &anglemapParams, Anglemap &anglemap,
                                                   const std::vector<size_t> &angleMapPDF,
                                                   double lookupValue) {
    //double lookupValue = randomGenerator.rnd();
    if (anglemapParams.phiWidth == 1) return -PI + 2.0 * PI * lookupValue; //special case, uniform phi distribution
    int phiLowerIndex;
    double weigh; //0: take previous theta line, 1: take next theta line, 0..1: interpolate in-between
    if (thetaLowerIndex == -1) { //first theta half section
        lookupValue += anglemap.phi_CDFs[0]; //periodic BCs over -PI...PI, can be larger than 1
        phiLowerIndex = my_lower_bound(lookupValue, &anglemap.phi_CDFs[0],
                                       anglemapParams.phiWidth); //take entirely the phi ditro belonging to first theta
        weigh = thetaOvershoot; // [0.5 - 1], will subtract 0.5 when evaluating thetaIndex
    } else if (thetaLowerIndex ==
               (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes - 1)) { //last theta half section
        lookupValue += anglemap.phi_CDFs[thetaLowerIndex *
                                         anglemapParams.phiWidth]; //periodic BCs over -PI...PI, can be larger than 1
        phiLowerIndex = my_lower_bound(lookupValue, &anglemap.phi_CDFs[thetaLowerIndex * anglemapParams.phiWidth],
                                       anglemapParams.phiWidth); //take entirely the phi ditro belonging to latest theta
        weigh = thetaOvershoot; // [0 - 0.5], will add 0.5 when evaluating thetaIndex
    } else {
        //Here we do a weighing both by the hit sum of the previous and next lines (w1 and w2) and also the weighs of the two lines based on thetaOvershoot (w3 and w4)
        // w1: sum of hits in previous line
        // w2: sum of hits in next line
        // w3: weigh of previous line (1 - thetaOvershoot)
        // w4: weigh of next line     (thetaOvershoot)
        // result: previous value weight: w1*w3 / (w1*w3 + w2*w4)
        //         next     value weight: w2*w4 / (w1*w3 + w2*w4) <- this will be the input for weighed_lower_bound

        double div;
        div = ((double) anglemap.phi_CDFsums[thetaLowerIndex] * (1.0 - thetaOvershoot) +
               (double) anglemap.phi_CDFsums[thetaLowerIndex + 1] * thetaOvershoot); // (w1*w3 + w2*w4)
        if (div > 0.0) {
            weigh = (thetaOvershoot * (double) anglemap.phi_CDFsums[thetaLowerIndex + 1]) /
                    div;    //      w2*w4 / (w1*w3 + w2*w4)
        } else {
            weigh = thetaOvershoot;
        }
        lookupValue += Weigh((double) anglemap.phi_CDFs[thetaLowerIndex * anglemapParams.phiWidth],
                             (double) anglemap.phi_CDFs[(thetaLowerIndex + 1) * anglemapParams.phiWidth], weigh);
        phiLowerIndex = weighed_lower_bound_X(lookupValue, weigh,
                                              &anglemap.phi_CDFs[thetaLowerIndex * anglemapParams.phiWidth],
                                              &anglemap.phi_CDFs[(thetaLowerIndex + 1) * anglemapParams.phiWidth],
                                              anglemapParams.phiWidth);
    }

    double phi, phiOvershoot;
    double thetaIndex = (double) thetaLowerIndex + 0.5 + weigh;
    if (phiLowerIndex == -1) { //first half section
        DEBUG_BREAK; //should not happen since we shifted the lookup value with first value
        phiOvershoot =
                0.5 + 0.5 * lookupValue / GetPhiCDFValue(thetaIndex, 0, anglemapParams, anglemap); //between 0.5 and 1
        phi = GetPhi((double) phiLowerIndex + 0.5 + phiOvershoot, anglemapParams); //between 0 and the first section end
    }
        /*else if (phiLowerIndex == (anglemapParams.phiWidth - 1)) { //last half section
            phiOvershoot = 0.5 * (lookupValue - GetPhiCDFValue(thetaIndex, phiLowerIndex, anglemapParams) )
                / (1.0 - GetPhiCDFValue(thetaIndex, phiLowerIndex, anglemapParams)); //between 0 and 0.5
            phi = GetPhi((double)phiLowerIndex + 0.5 + phiOvershoot, anglemapParams); //between 0 and the first section end
        }*/
    else { //regular or last section
        if (/*true ||*/ GetPhipdfValue(thetaIndex, phiLowerIndex, anglemapParams, angleMapPDF) ==
                        GetPhipdfValue(thetaIndex, phiLowerIndex + 1, anglemapParams, angleMapPDF)) {
            //The pdf's slope is 0, linear interpolation
            phiOvershoot = (lookupValue - GetPhiCDFValue(thetaIndex, phiLowerIndex, anglemapParams, anglemap))
                           / (GetPhiCDFValue(thetaIndex, phiLowerIndex + 1, anglemapParams, anglemap) -
                              GetPhiCDFValue(thetaIndex, phiLowerIndex, anglemapParams, anglemap));
            phi = GetPhi((double) phiLowerIndex + 0.5 + phiOvershoot, anglemapParams);
        } else {

            //2nd degree interpolation
            // y(x) = ax^2 + bx + c
            // c: CDF value at lower index
            // b: pdf value at lower index
            // a: pdf slope at lower index / 2
            // dy := y - c
            // dx := x - [x at lower index]
            // dy = ax^2 + bx
            // dx = ( -b + sqrt(b^2 +4*a*dy) ) / (2a)
            double phiStep = 2.0 * PI / (double) anglemapParams.phiWidth;
            double c = GetPhiCDFValue(thetaIndex, phiLowerIndex, anglemapParams, anglemap); //CDF value at lower index
            double b = GetPhipdfValue(thetaIndex, phiLowerIndex, anglemapParams, angleMapPDF) /
                       GetPhiCDFSum(thetaIndex, anglemapParams, anglemap) / phiStep; //pdf value at lower index
            double a = 0.5 * (GetPhipdfValue(thetaIndex, phiLowerIndex + 1, anglemapParams, angleMapPDF) -
                              GetPhipdfValue(thetaIndex, phiLowerIndex, anglemapParams, angleMapPDF)) /
                       GetPhiCDFSum(thetaIndex, anglemapParams, anglemap) / Sqr(phiStep); //pdf slope at lower index
            double dy = lookupValue - c;

            double D = Sqr(b) + 4 * a *
                                dy; //Discriminant. In rare cases it might be slightly negative, then fall back to linear interpolation:
            if (D < 0) {
                phiOvershoot = (lookupValue - GetPhiCDFValue(thetaIndex, phiLowerIndex, anglemapParams,
                                                             anglemap))
                               / (GetPhiCDFValue(thetaIndex, (int) IDX(phiLowerIndex + 1, anglemapParams.phiWidth),
                                                 anglemapParams, anglemap) -
                                  GetPhiCDFValue(thetaIndex, phiLowerIndex, anglemapParams, anglemap));
            } else {
                double dx = (-b + sqrt(Sqr(b) + 4 * a * dy)) /
                            (2 * a); //Since b>=0 it's the + branch of the +- that is valid for us
                phiOvershoot = dx / phiStep;
            }
            phi = GetPhi((double) phiLowerIndex + 0.5 + phiOvershoot, anglemapParams);
        }
    }
    assert(phi > -PI && phi < PI);
    return phi;
}

/**
* \brief Converts from index to theta value
* \param thetaIndex theta index
* \param anglemapParams parameters of the angle map
* \return theta angle
*/

double AnglemapGeneration::GetTheta(const double &thetaIndex, const AnglemapParams &anglemapParams) {
    if ((size_t) (thetaIndex) < anglemapParams.thetaLowerRes) { // 0 < theta < limit
        return anglemapParams.thetaLimit * (thetaIndex) / (double) anglemapParams.thetaLowerRes;
    } else { // limit < theta < PI/2
        return anglemapParams.thetaLimit +
               (PI / 2.0 - anglemapParams.thetaLimit) * (thetaIndex - (double) anglemapParams.thetaLowerRes) /
               (double) anglemapParams.thetaHigherRes;
    }
}

/**
* \brief makes phiIndex circular and converts from index to -pi...pi
* \param phiIndex phi index
* \param anglemapParams parameters of the angle map
* \return phi angle
*/

double AnglemapGeneration::GetPhi(const double &phiIndex, const AnglemapParams &anglemapParams)
//makes phiIndex circular and converts from index to -pi...pi
{
    double width = (double) anglemapParams.phiWidth;
    double correctedIndex = (phiIndex < width) ? phiIndex : phiIndex - width;
    return -PI + 2.0 * PI * correctedIndex / width;
}

double
AnglemapGeneration::GetPhipdfValue(const double &thetaIndex, const int &phiLowerIndex,
                                   const AnglemapParams &anglemapParams, const std::vector<size_t> &angleMapPDF)
//phiLowerIndex is circularized
{
    if (thetaIndex < 0.5) {
        return (double) angleMapPDF[IDX(phiLowerIndex, anglemapParams.phiWidth)];
    } else if (thetaIndex > (double) (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes) - 0.5) {
        return (double) angleMapPDF[
                anglemapParams.phiWidth * (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes - 1) +
                IDX(phiLowerIndex, anglemapParams.phiWidth)];
    } else {
        size_t thetaLowerIndex = (size_t) (thetaIndex - 0.5);
        double thetaOvershoot = thetaIndex - 0.5 - (double) thetaLowerIndex;
        double valueFromLowerpdf = (double) angleMapPDF[anglemapParams.phiWidth * thetaLowerIndex +
                                                        IDX(phiLowerIndex, anglemapParams.phiWidth)];
        double valueFromHigherpdf = (double) angleMapPDF[anglemapParams.phiWidth * (thetaLowerIndex + 1) +
                                                         IDX(phiLowerIndex, anglemapParams.phiWidth)];
        return Weigh(valueFromLowerpdf, valueFromHigherpdf, thetaOvershoot);
    }
}

/**
* \brief Get phi value from cummulative density function
* \param thetaIndex theta index
* \param phiLowerIndex lower index of the bin
* \param anglemapParams parameters of the angle map
* \return phi cdf value
*/

double
AnglemapGeneration::GetPhiCDFValue(const double &thetaIndex, const int &phiLowerIndex,
                                   const AnglemapParams &anglemapParams, const Anglemap &anglemap) {
    if (thetaIndex < 0.5) {
        return (phiLowerIndex < anglemapParams.phiWidth) ? anglemap.phi_CDFs[phiLowerIndex] : 1.0 +
                                                                                              anglemap.phi_CDFs[0];
    } else if (thetaIndex > (double) (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes) - 0.5) {
        return (phiLowerIndex < anglemapParams.phiWidth) ? anglemap.phi_CDFs[
                anglemapParams.phiWidth * (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes - 1) +
                phiLowerIndex] : 1.0 + anglemap.phi_CDFs[anglemapParams.phiWidth *
                                                         (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes -
                                                          1)];
    } else {
        size_t thetaLowerIndex = (size_t) (thetaIndex - 0.5);
        double thetaOvershoot = thetaIndex - 0.5 - (double) thetaLowerIndex;
        double valueFromLowerCDF = (phiLowerIndex < anglemapParams.phiWidth) ? anglemap.phi_CDFs[
                anglemapParams.phiWidth * thetaLowerIndex + phiLowerIndex] : 1.0 +
                                                                             anglemap.phi_CDFs[anglemapParams.phiWidth *
                                                                                               (thetaLowerIndex)];
        double valueFromHigherCDF = (phiLowerIndex < anglemapParams.phiWidth) ? anglemap.phi_CDFs[
                anglemapParams.phiWidth * (thetaLowerIndex + 1) + phiLowerIndex] : 1.0 +
                                                                                   anglemap.phi_CDFs[
                                                                                           anglemapParams.phiWidth *
                                                                                           (thetaLowerIndex + 1)];
        return Weigh(valueFromLowerCDF, valueFromHigherCDF, thetaOvershoot);
    }

}

/**
* \brief Get phi value from cummulative density function
* \param thetaIndex theta index
* \param anglemapParams parameters of the angle map
* \return phi cdf summed value
*/

double AnglemapGeneration::GetPhiCDFSum(const double &thetaIndex, const AnglemapParams &anglemapParams,
                                        const Anglemap &anglemap) {
    if (thetaIndex < 0.5) {
        return (double) anglemap.phi_CDFsums[0];
    } else if (thetaIndex > (double) (anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes) - 0.5) {
        return (double) anglemap.phi_CDFsums[anglemapParams.thetaLowerRes + anglemapParams.thetaHigherRes - 1];
    } else {
        size_t thetaLowerIndex = (size_t) (thetaIndex - 0.5);
        double thetaOvershoot = thetaIndex - 0.5 - (double) thetaLowerIndex;
        double valueFromLowerSum = (double) anglemap.phi_CDFsums[thetaLowerIndex];
        double valueFromHigherSum = (double) anglemap.phi_CDFsums[thetaLowerIndex + 1];
        return Weigh(valueFromLowerSum, valueFromHigherSum, thetaOvershoot);
    }
}