package Families.impl;

import Families.FamilyMember;
import Families.Family;
import Families.FamilyRegister;
import Families.FamiliesFactory;
import Families.FamiliesPackage;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the factory '<em><b>Families Factory</b></em>'.
 * <!-- end-user-doc -->
 *
 * @generated
 */
public class FamiliesFactoryImpl extends MinimalEObjectImpl implements FamiliesFactory {
	/**
	 * @generated
	 */
	public static FamiliesFactory init() {
		return theFamiliesFactory;
	}

	/**
	 * @generated
	 */
	private static FamiliesFactory theFamiliesFactory;

	/**
	 * @generated
	 */
	protected FamiliesFactoryImpl() {
		super();
	}

	/**
	 * @generated
	 */
	@Override
	public EObject create(EClass eClass) {
		switch (eClass.getClassifierID()) {
			case FamiliesPackageImpl.Literals.FAMILY_REGISTER:
				return createFamilyRegister();
			case FamiliesPackageImpl.Literals.FAMILY:
				return createFamily();
			case FamiliesPackageImpl.Literals.FAMILY_MEMBER:
				return createFamilyMember();
			default:
				throw new IllegalArgumentException("The class '" + eClass.getName() + "' is not a valid classifier");
		}
	}

	/**
	 * @generated
	 */
	@Override
	public FamilyRegister createFamilyRegister() {
		FamilyRegisterImpl familyRegister = new FamilyRegisterImpl();
		return familyRegister;
	}

	/**
	 * @generated
	 */
	@Override
	public Family createFamily() {
		FamilyImpl family = new FamilyImpl();
		return family;
	}

	/**
	 * @generated
	 */
	@Override
	public FamilyMember createFamilyMember() {
		FamilyMemberImpl familyMember = new FamilyMemberImpl();
		return familyMember;
	}

	/**
	 * @generated
	 */
	@Override
	public FamiliesPackage getFamiliesPackage() {
		return FamiliesPackageImpl.init();
	}
} // FamiliesFactoryImpl