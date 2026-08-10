package Persons.impl;

import Persons.Person;
import Persons.PersonRegister;

import java.util.Calendar;
import java.util.Date;

/**
 * A minimal stub implementation of Person for transformation testing.
 * 
 * @generated NOT
 */
public class PersonImpl implements Person {
	
	private String name;
	private Date birthday;
	private PersonRegister container;
	
	public PersonImpl() {
		// Default constructor
		setDefaultBirthday();
	}
	
	private void setDefaultBirthday() {
		Calendar cal = Calendar.getInstance();
		cal.set(0, 0, 1); // Year 0, January 1st
		this.birthday = cal.getTime();
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
	public Date getBirthday() {
		return birthday;
	}
	
	@Override
	public void setBirthday(Date birthday) {
		this.birthday = birthday;
	}
	
	@Override
	public PersonRegister getPersonsInverse() {
		return container;
	}
	
	@Override
	public void setPersonsInverse(PersonRegister register) {
		this.container = register;
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
} // PersonImpl