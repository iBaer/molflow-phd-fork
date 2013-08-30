/*
  File:        PressureEvolution.h
  Description: Pressure Evolution plotter window
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
#include "GLApp/GLChart.h"
#include "GLApp/GLLabel.h"
#include "GLApp/GLCombo.h"
#include "GLApp/GLButton.h"
#include "GLApp/GLParser.h"
#include "GLApp/GLTextField.h"
#include "Worker.h"
#include "Geometry.h"

#ifndef _PRESSUREEVOLUTIONH_
#define _PRESSUREEVOLUTIONH_

#define MODE_SLICE //display selected slice
#define MODE_SUMAVG //sum or average mode

class PressureEvolution : public GLWindow {

public:

  // Construction
  PressureEvolution();

  // Component method
  void Display(Worker *w);
  void Refresh();
  void Update(float appTime,BOOL force=FALSE);
  void Reset();

  // Implementation
  void ProcessMessage(GLComponent *src,int message);
  void SetBounds(int x,int y,int w,int h);

private:

  void addView(int facet);
  void remView(int facet);
  void refreshViews();
  void plot();

  Worker      *worker;
  GLButton    *dismissButton;
  GLChart     *chart;
  GLCombo     *profCombo;
  GLLabel     *normLabel;
  //GLLabel	  *qLabel;
  //GLLabel     *unitLabel;
  GLLabel     *label1;
  GLCombo     *normCombo;
  //GLToggle    *showAllMoments;
  GLButton    *selButton;
  GLButton    *addButton;
  GLButton    *removeButton;
  GLButton    *resetButton;
  GLButton    *setSliceButton;
  GLTextField *formulaText;
  //GLTextField *qText;
  GLButton    *formulaBtn;

  GLCombo *modeCombo;
  GLTextField *selectedSliceText;

  GLToggle *logXToggle,*logYToggle;

  GLDataView  *views[32];

  int          nbView;
  int selectedSlice;
  float        lastUpdate;

};

#endif /* _PRESSUREEVOLUTIONH_ */
