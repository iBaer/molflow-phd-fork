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

#include "InterfaceXML.h"
#include <MolFlow.h>
#include <ProfilePlotter.h>
#include <ConvergencePlotter.h>
#include <versionId.h>
#include <sstream>
#include <iomanip>

namespace FlowIO {
    using namespace pugi;

    void LoaderInterfaceXML::LoadInterface(xml_node interfNode, MolFlow *mApp) {
        //xml_node interfNode = loadXML.child("Interface");

        xml_node selNode = interfNode.child("Selections");
        //int nbS = selNode.select_nodes("Selection").size();

        for (xml_node sNode : selNode.children("Selection")) {
            SelectionGroup s;
            s.name = sNode.attribute("name").as_string();
            s.selection.reserve(sNode.select_nodes("selItem").size());
            for (xml_node iNode : sNode.children("selItem"))
                s.selection.push_back(iNode.attribute("facet").as_llong());
            mApp->AddSelection(s);
        }

        xml_node viewNode = interfNode.child("Views");
        for (xml_node newView : viewNode.children("View")) {
            AVIEW v;
            v.name = newView.attribute("name").as_string();
            v.projMode = newView.attribute("projMode").as_int();
            v.camAngleOx = newView.attribute("camAngleOx").as_double();
            v.camAngleOy = newView.attribute("camAngleOy").as_double();
            if (newView.attribute("camAngleOz")) {
                v.camAngleOz = newView.attribute("camAngleOz").as_double();
            }
            else {
                v.camAngleOz = 0.0; //Otherwise RoundAngle() routine hangs for unitialized value
            }
            if (newView.attribute("lightAngleOx")) {
                v.lightAngleOx = newView.attribute("lightAngleOx").as_double();
            }
            else {
                v.lightAngleOx = 0.0;
            }
            if (newView.attribute("lightAngleOy")) {
                v.lightAngleOy = newView.attribute("lightAngleOy").as_double();
            }
            else {
                v.lightAngleOy = 0.0;
            }
            v.camDist = newView.attribute("camDist").as_double();
            v.camOffset.x = newView.attribute("camOffset.x").as_double();
            v.camOffset.y = newView.attribute("camOffset.y").as_double();
            v.camOffset.z = newView.attribute("camOffset.z").as_double();
            v.performXY = newView.attribute("performXY").as_int();
            v.vLeft = newView.attribute("vLeft").as_double();
            v.vRight = newView.attribute("vRight").as_double();
            v.vTop = newView.attribute("vTop").as_double();
            v.vBottom = newView.attribute("vBottom").as_double();
            mApp->AddView(v.name.c_str(), v);

        }

        xml_node formulaNode = interfNode.child("Formulas");
        if(formulaNode) {
            for (xml_node newFormula : formulaNode.children("Formula")) {
                mApp->AddFormula(newFormula.attribute("name").as_string(),
                                 newFormula.attribute("expression").as_string());
            }
        }
        xml_node ppNode = interfNode.child("ProfilePlotter");
        if (ppNode) {
            if (!mApp->profilePlotter) mApp->profilePlotter = new ProfilePlotter();
            mApp->profilePlotter->SetWorker(&mApp->worker);
            xml_node paramsNode = ppNode.child("Parameters");
            if (paramsNode && paramsNode.attribute("logScale"))
                mApp->profilePlotter->SetLogScaled(paramsNode.attribute("logScale").as_bool());
            xml_node viewsNode = ppNode.child("Views");
            if (viewsNode) {
                std::vector<int> views;
                for (xml_node view : viewsNode.children("View"))
                    views.push_back(view.attribute("facetId").as_int());
                mApp->profilePlotter->SetViews(views);
            }
        }

        xml_node cpNode = interfNode.child("ConvergencePlotter");
        if (cpNode) {
            if (!mApp->convergencePlotter) {
                mApp->convergencePlotter = new ConvergencePlotter(&mApp->worker, mApp->formula_ptr);
                mApp->convergencePlotter->SetWorker(&mApp->worker);
            }
            xml_node paramsNode = cpNode.child("Parameters");
            if (paramsNode && paramsNode.attribute("logScale"))
                mApp->convergencePlotter->SetLogScaled(paramsNode.attribute("logScale").as_bool());
            xml_node viewsNode = cpNode.child("Views");
            if (viewsNode) {
                std::vector<int> views;
                for (xml_node view : viewsNode.children("View"))
                    views.push_back(view.attribute("formulaHash").as_int());
                mApp->convergencePlotter->SetViews(views);
            }
        }
        // return;
    }

    void WriterInterfaceXML::WriteInterface(xml_document &saveDoc, MolFlow *mApp, bool saveSelected) {

        xml_node rootNode;
        if(mApp->useOldXMLFormat){
            rootNode = saveDoc.root();
        }
        else {
            rootNode = saveDoc.child("SimulationEnvironment");
            if(!rootNode) {
                rootNode = saveDoc.append_child("SimulationEnvironment");
                rootNode.attribute("type") = "molflow";
                rootNode.append_attribute("version") = appVersionId;
            }
        }

        xml_node interfNode = rootNode.append_child("Interface");

        xml_node selNode = interfNode.append_child("Selections");
        selNode.append_attribute("nb") = (!saveSelected) * (mApp->selections.size());
        for (size_t i = 0; (i < mApp->selections.size()) &&
                           !saveSelected; i++) { //don't save selections when exporting part of the geometry (saveSelected)
            xml_node newSel = selNode.append_child("Selection");
            newSel.append_attribute("id") = i;
            newSel.append_attribute("name") = mApp->selections[i].name.c_str();
            newSel.append_attribute("nb") = mApp->selections[i].selection.size();
            for (size_t j = 0; j < mApp->selections[i].selection.size(); j++) {
                xml_node newItem = newSel.append_child("selItem");
                newItem.append_attribute("id") = j;
                newItem.append_attribute("facet") = mApp->selections[i].selection[j];
            }
        }

        xml_node viewNode = interfNode.append_child("Views");
        viewNode.append_attribute("nb") = (!saveSelected) * (mApp->nbView);
        for (int i = 0; (i < mApp->nbView) &&
                        !saveSelected; i++) { //don't save views when exporting part of the geometry (saveSelected)
            xml_node newView = viewNode.append_child("View");
            newView.append_attribute("id") = i;
            newView.append_attribute("name") = mApp->views[i].name.c_str();
            newView.append_attribute("projMode") = mApp->views[i].projMode;
            newView.append_attribute("camAngleOx") = mApp->views[i].camAngleOx;
            newView.append_attribute("camAngleOy") = mApp->views[i].camAngleOy;
            newView.append_attribute("camAngleOz") = mApp->views[i].camAngleOz;
            newView.append_attribute("camDist") = mApp->views[i].camDist;
            newView.append_attribute("lightAngleOx") = mApp->views[i].lightAngleOx;
            newView.append_attribute("lightAngleOy") = mApp->views[i].lightAngleOy;
            newView.append_attribute("camOffset.x") = mApp->views[i].camOffset.x;
            newView.append_attribute("camOffset.y") = mApp->views[i].camOffset.y;
            newView.append_attribute("camOffset.z") = mApp->views[i].camOffset.z;
            newView.append_attribute("performXY") = mApp->views[i].performXY;
            newView.append_attribute("vLeft") = mApp->views[i].vLeft;
            newView.append_attribute("vRight") = mApp->views[i].vRight;
            newView.append_attribute("vTop") = mApp->views[i].vTop;
            newView.append_attribute("vBottom") = mApp->views[i].vBottom;
        }

        xml_node formulaNode = interfNode.append_child("Formulas");
        formulaNode.append_attribute("nb") = (!saveSelected) * (mApp->formula_ptr->formulas_n->size());
        if (!saveSelected) { //don't save formulas when exporting part of the geometry (saveSelected)
            for (size_t i = 0; i < mApp->formula_ptr->formulas_n->size(); i++) {
                xml_node newFormula = formulaNode.append_child("Formula");
                newFormula.append_attribute("id") = i;
                newFormula.append_attribute("name") = mApp->formula_ptr->formulas_n->at(i)->GetName();
                newFormula.append_attribute("expression") = mApp->formula_ptr->formulas_n->at(i)->GetExpression();
            }
        }

        if (mApp->profilePlotter) {
            std::vector<int> ppViews = mApp->profilePlotter->GetViews();
            xml_node profilePlotterNode = interfNode.append_child("ProfilePlotter");
            profilePlotterNode.append_child("Parameters").append_attribute(
                    "logScale") = (int) mApp->profilePlotter->IsLogScaled(); //backward compatibility: 0 or 1
            xml_node viewsNode = profilePlotterNode.append_child("Views");
            for (int v : ppViews) {
                xml_node view = viewsNode.append_child("View");
                view.append_attribute("facetId") = v;
            }
        }

        if (mApp->convergencePlotter) {
            std::vector<int> cpViews = mApp->convergencePlotter->GetViews();
            xml_node convergencePlotterNode = interfNode.append_child("ConvergencePlotter");
            convergencePlotterNode.append_child("Parameters").append_attribute(
                    "logScale") = (int) mApp->convergencePlotter->IsLogScaled(); //backward compatibility: 0 or 1
            xml_node viewsNode = convergencePlotterNode.append_child("Views");
            for (int v : cpViews) {
                xml_node view = viewsNode.append_child("View");
                view.append_attribute("formulaHash") = v;
            }
        }


        // TODO: Move to other place once convergence is part of CLI
        xml_node resultNode = rootNode.child("MolflowResults");
        if(!resultNode) {
            resultNode = rootNode.append_child("MolflowResults");
        }
        //Convergence results
        xml_node convNode = resultNode.append_child("Convergence");

        int formulaId = 0;
        for(const auto& formulaVec : mApp->formula_ptr->convergenceValues){
            std::stringstream convText;
            convText << std::setprecision(10) << '\n';
            convText << std::scientific;
            for(const auto& convVal : formulaVec.conv_vec){
                convText << convVal.first << "\t" << convVal.second << "\n";
            }
            xml_node newFormulaNode = convNode.append_child("ConvData");
            newFormulaNode.append_attribute("Formula") = mApp->formula_ptr->formulas_n->at(formulaId)->GetExpression();
            xml_node newConv = newFormulaNode.append_child(node_cdata);
            newConv.set_value(convText.str().c_str());
            formulaId++;
        }
    }
}