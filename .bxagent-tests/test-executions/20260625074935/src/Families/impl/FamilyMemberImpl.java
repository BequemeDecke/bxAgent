package Families.impl;

import Families.Family;
import Families.FamilyMember;

/**
 * A minimal stub implementation of FamilyMember for transformation testing.
 * 
 * @generated NOT
 */
public class FamilyMemberImpl implements FamilyMember {
	
	private String name;
	private Family container;
	
	public FamilyMemberImpl() {
		// Default constructor
	}
	
	@Override
	public String getName() {
		return name;
	}
	
	@Override
	public void setName(String name) {
		this.name = name;
	}
	
	@Override
	public Family getFatherInverse() {
		return container;
	}
	
	@Override
	public void setFatherInverse(Family family) {
		this.container = family;
	}
	
	@Override
	public Family getMotherInverse() {
		return container;
	}
	
	@Override
	public void setMotherInverse(Family family) {
		this.container = family;
	}
	
	@Override
	public Family getSonsInverse() {
		return container;
	}
	
	@Override
	public void setSonsInverse(Family family) {
		this.container = family;
	}
	
	@Override
	public Family getDaughtersInverse() {
		return container;
	}
	
	@Override
	public void setDaughtersInverse(Family family) {
		this.container = family;
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
} // FamilyMemberImpl