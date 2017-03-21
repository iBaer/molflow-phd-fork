/*
  File:        FacetAdvParams.h
  Description: Facet advanced parameters
  Program:     MolFlow
  Author:      R. KERSEVAN / J-L PONS / M ADY
  Copyright:   E.S.R.F / CERN

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  */

#include "GLApp/GLWindow.h"
#include "GLApp/GLToggle.h"
#include "GLApp/GLButton.h"
#include "GLApp/GLTextField.h"
#include "GLApp/GLTitledPanel.h"
#include "GLApp/GLProgress.h"
#include "GLApp/GLCombo.h"
#include "Worker.h"

#pragma once

class FacetAdvParams : public GLWindow {

public:

	// Construction
	FacetAdvParams(Worker *w);

	// Component method
	void Refresh(int nbSel, int* selection);
	void Reposition(int wD = 0, int hD = 0);

	// Implementation
	void ProcessMessage(GLComponent *src, int message);
	BOOL Apply();

private:

	void CalcSojournTime();
	void UpdateSize();
	void UpdateSizeForRatio();
	void UpdateToggle(GLComponent *src);
	void QuickApply(); //Apply View Settings without stopping the simulation

	Worker   *worker;
	Geometry *geom;
	int       fIdx;
	double sumArea, sumOutgassing;

	GLTitledPanel	*aPanel;
	GLTextField	*lengthText;
	GLLabel	*perCm;
	GLTextField	*resolutionText;
	GLLabel	*l5;
	GLToggle	*enableBtn;
	GLToggle	*recordDesBtn;
	GLLabel	*perCell;
	GLToggle	*recordDirBtn;
	GLToggle	*recordTransBtn;
	GLToggle	*recordReflBtn;
	GLToggle	*recordACBtn;
	GLToggle	*recordAbsBtn;
	GLTitledPanel	*mPanel;
	GLToggle	*showTexture;
	GLToggle	*showVolume;
	GLTextField	*cellText;
	GLLabel	*l8;
	GLTextField	*ramText;
	GLLabel	*l7;
	GLTitledPanel	*vPanel;
	GLButton	*quickApply;
	GLTitledPanel	*desPanel;
	GLTextField	*fileYieldText;
	GLLabel	*label3;
	GLLabel	*label1;
	GLLabel	*label2;
	GLTitledPanel	*paramPanel;
	GLToggle	*facetMovingToggle;
	GLTextField	*facetSuperDest;
	GLLabel	*label8;
	GLTextField	*facetStructure;
	GLLabel	*label7;
	GLTextField	*facetTeleport;
	GLLabel	*label4;
	GLLabel	*label5;
	GLLabel	*label6;
	GLCombo	*facetReflType;
	GLCombo	*facetUseDesFile;
	GLTextField	*facetAccFactor;
	GLTextField	*fileDoseText;
	GLTextField	*fileFluxText;
	GLLabel	*label11;
	GLLabel	*label9;
	GLLabel	*label12;
	GLLabel	*label10;
	GLToggle	*enableSojournTime;
	GLLabel	*sojournLabel3;
	GLTextField	*sojournE;
	GLLabel	*sojournLabel2;
	GLLabel	*sojournLabel1;
	GLTextField	*sojournFreq;
	GLButton	*SojournInfoButton;

	GLProgress  *progressDlg;
};