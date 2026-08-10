package Persons.impl;

import Persons.PersonsPackage;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EPackage;

/**
 * A minimal stub implementation of PersonsPackage for transformation testing.
 * 
 * @generated NOT
 */
public class PersonsPackageImpl implements PersonsPackage {
	
	private static PersonsPackageImpl instance;
	
	public static PersonsPackageImpl getInstance() {
		if (instance == null) {
			instance = new PersonsPackageImpl();
		}
		return instance;
	}
	
	@Override
	public EClass getPersonRegister() {
		return null;
	}
	
	@Override
	public EClass getPerson() {
		return null;
	}
	
	@Override
	public EClass getMale() {
		return null;
	}
	
	@Override
	public EClass getFemale() {
		return null;
	}
	
	@Override
	public String getName() {
		return "Persons";
	}
	
	@Override
	public String getNsURI() {
		return "http://Persons";
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
	public PersonsPackage init() {
		return this;
	}
} // PersonsPackageImpl