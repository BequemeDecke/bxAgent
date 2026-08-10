package Persons.impl;

import Persons.Person;
import Persons.PersonRegister;

import java.util.ArrayList;
import java.util.List;

/**
 * A minimal stub implementation of PersonRegister for transformation testing.
 * 
 * @generated NOT
 */
public class PersonRegisterImpl implements PersonRegister {
	
	private List<Person> persons = new ArrayList<>();
	
	public PersonRegisterImpl() {
		// Default constructor
	}
	
	@Override
	public List<Person> getPersons() {
		return persons;
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
} // PersonRegisterImpl