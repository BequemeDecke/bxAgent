package Families.impl;

import Families.Family;
import Families.FamilyMember;
import Families.FamilyRegister;
import Families.FamiliesFactory;
import Families.FamiliesPackage;

/**
 * <!-- begin-user-doc -->
 * The <b>Factory</b> for the model.
 * It provides a create method for each non-abstract class of the model.
 * <!-- end-user-doc -->
 * @generated
 */
public class FamiliesFactoryImpl implements FamiliesFactory {
	/**
	 * The singleton instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static FamiliesFactoryImpl instance;

	/**
	 * Creates the default factory implementation.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static FamiliesFactory init() {
		if (instance == null) {
			instance = new FamiliesFactoryImpl();
		}
		return instance;
	}

	/**
	 * Creates an instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public FamiliesFactoryImpl() {
		super();
	}

	@Override
	public FamilyRegister createFamilyRegister() {
		return new FamilyRegisterImpl();
	}

	@Override
	public Family createFamily() {
		return new FamilyImpl();
	}

	@Override
	public FamilyMember createFamilyMember() {
		return new FamilyMemberImpl();
	}

	@Override
	public FamiliesPackage getFamiliesPackage() {
		return FamiliesPackageImpl.getInstance();
	}
} // FamiliesFactoryImpl