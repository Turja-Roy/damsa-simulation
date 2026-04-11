#ifndef MAGNETICFIELD_H
#define MAGNETICFIELD_H

#include "G4MagneticField.hh"
#include "G4SystemOfUnits.hh"

class MagneticField : public G4MagneticField
{
public:
    MagneticField();
    virtual ~MagneticField();
    
    virtual void GetFieldValue(const G4double point[4], G4double* bField) const override;

private:
    G4double fFieldStrength;
};

inline MagneticField::MagneticField()
: G4MagneticField(), fFieldStrength(3.0*tesla)
{}

inline MagneticField::~MagneticField()
{}

inline void MagneticField::GetFieldValue(const G4double point[4], G4double* bField) const
{
    G4double x = point[0];
    G4double y = point[1];
    
    bField[0] = 0.;
    bField[1] = 0.;
    bField[2] = 0.;
    
    if (x > 0) {
        bField[0] = -fFieldStrength;
    } else if (x < 0) {
        bField[0] = fFieldStrength;
    }
    
    if (y > 0) {
        bField[1] = -fFieldStrength;
    } else if (y < 0) {
        bField[1] = fFieldStrength;
    }
}

#endif