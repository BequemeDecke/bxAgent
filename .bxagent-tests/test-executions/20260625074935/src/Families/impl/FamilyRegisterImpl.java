package Families.impl;

import Families.Family;
import Families.FamilyRegister;

import java.util.ArrayList;
import java.util.List;

/**
 * A minimal stub implementation of FamilyRegister for transformation testing.
 * 
 * @generated NOT
 */
public class FamilyRegisterImpl implements FamilyRegister {
	
	private List<Family> families = new ArrayList<>();
	
	public FamilyRegisterImpl() {
		// Default constructor
	}
	
	@Override
	public List<Family> getFamilies() {
		return families;
	}
	
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		return null;
	}
	
	public void eSet(int featureID, Object newValue) {
		// No-op for stub
	}
	
	public boolean eIsSet(int featureID) {
		return false;
	}
	
	public Object eInvoke(int operationID, Object[] arguments) {
		return null;
	}
} // FamilyRegisterImpl