package Families.impl;

import Families.FamiliesPackage;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EPackage;

/**
 * A minimal stub implementation of FamiliesPackage for transformation testing.
 * 
 * @generated NOT
 */
public class FamiliesPackageImpl implements FamiliesPackage {
	
	private static FamiliesPackageImpl instance;
	
	public static FamiliesPackageImpl getInstance() {
		if (instance == null) {
			instance = new FamiliesPackageImpl();
		}
		return instance;
	}
	
	@Override
	public EClass getFamilyRegister() {
		return null;
	}
	
	@Override
	public EClass getFamily() {
		return null;
	}
	
	@Override
	public EClass getFamilyMember() {
		return null;
	}
	
	@Override
	public String getName() {
		return "Families";
	}
	
	@Override
	public String getNsURI() {
		return "http://Families";
	}
	
	@Override
	public EPackage getParent() {
		return null;
	}
	
	@Override
	public void setName(String name) {
		// No-op for stub
	}
	
	@Override
	public void setNsURI(String uri) {
		// No-op for stub
	}
	
	@Override
	public FamiliesPackage init() {
		return this;
	}
} // FamiliesPackageImpl