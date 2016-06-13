/*
  File:        GLTextArea.h
  Description: Text fiedl class (SDL/OpenGL OpenGL application framework)
  Author:      Marton ADY (2013)

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
*/
#include "GLComponent.h"

#ifndef _GLTEXTAREAH_
#define _GLTEXTAREAH_

#define MAX_TEXTAREA_SIZE 65536

class GLTextArea : public GLComponent {

public:

  // Construction
  GLTextArea(int compId,char *text);

  // Component methods
  void SetText(char *text);
  char *GetText();
  void ScrollToVisible();
  void SetCursorPos(int pos);
  int  GetCursorPos();
  int  GetTextLength();
  void SetEditable(BOOL editable);
  void SetEditable_NoBG(BOOL editable);
  void Clear();
  // ------------------------------------------------------


  BOOL GetNumber(double *num);
  BOOL GetNumberInt(int *num);
  void SelectAll();
  BOOL IsCaptured();

  // Implementation
  void Paint();
  void ManageEvent(SDL_Event *evt);
  void SetFocus(BOOL focus);

private:

  void   CopyClipboardText();
  void   PasteClipboardText();
  void   UpdateXpos();
  void   InsertString(char *lpszString);
  void   DeleteString(int count);
  void   MoveCursor(int newPos);
  void   RemoveSel();
  void   MoveSel(int newPos);
  void   DeleteSel();
  void   UpdateText(char *text);
  void	 ProcessEnter();
  int    GetCursorLocation(int px);

  char    m_Text[MAX_TEXTAREA_SIZE];
  int      m_Start;
  int      m_Stop;
  int      m_Length;
  int      m_CursorPos;
  short    m_XPos[MAX_TEXTAREA_SIZE];
  int      m_CursorState;
  int      m_Captured;
  int      m_LastPos;
  int     m_Zero;
  BOOL    m_Editable;

};

#endif /* _GLTEXTAREAH_ */