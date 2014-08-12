// Copyright (c) George Washington University, all rights reserved.  See the
// accompanying LICENSE file for more information.

#pragma once

#include <map>
#include <vector>

typedef std::vector< std::vector<double> > Palette;

/////////////////////////////////////////////////
enum PaletteType {
  eOriginalPalette = 1,
  eNewPalette      = 2,
  eOblivion        = 3
};

//////////////////////////////////////////////////
class ColorPalette {

 public:

  ColorPalette();
  ~ColorPalette();
  Palette  GetPalette( PaletteType type );
  Palette& GetPaletteRef( PaletteType type );

 private:
  void _InitColorPalettes();

 private:
  std::map< PaletteType, Palette > m_mPalettes;

};
