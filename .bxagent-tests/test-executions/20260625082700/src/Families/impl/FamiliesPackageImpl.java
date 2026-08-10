package Families.impl;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;
import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.impl.MinimalEObjectImpl;
import org.eclipse.emf.ecore.impl.PackageImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the package '<em><b>Families Package</b></em>'.
 * <!-- end-user-doc -->
 *
 * @generated
 */
public class FamiliesPackageImpl extends PackageImpl implements FamiliesPackage {
	/**
	 * @generated
	 */
	public static final int FAMILY_REGISTER__FAMILIES = 0;

	/**
	 * @generated
	 */
	public static final int FAMILY_REGISTER_FEATURE_COUNT = 1;

	/**
	 * @generated
	 */
	public static final int FAMILY_REGISTER__NAME = FAMILY_REGISTER_FEATURE_COUNT + 0;

	/**
	 * @generated
	 */
	public static final int FAMILY__FATHER = 0;

	/**
	 * @generated
	 */
	public static final int FAMILY__MOTHER = 1;

	/**
	 * @generated
	 */
	public static final int FAMILY__SONS = 2;

	/**
	 * @generated
	 */
	public static final int FAMILY__DAUGHTERS = 3;

	/**
	 * @generated
	 */
	public static final int FAMILY__NAME = 4;

	/**
	 * @generated
	 */
	public static final int FAMILY__FAMILIES_INVERSE = 5;

	/**
	 * @generated
	 */
	public static final int FAMILY_FEATURE_COUNT = 6;

	/**
	 * @generated
	 */
	public static final int FAMILY_MEMBER__NAME = 0;

	/**
	 * @generated
	 */
	public static final int FAMILY_MEMBER__FATHER_INVERSE = 1;

	/**
	 * @generated
	 */
	public static final int FAMILY_MEMBER__MOTHER_INVERSE = 2;

	/**
	 * @generated
	 */
	public static final int FAMILY_MEMBER__SONS_INVERSE = 3;

	/**
	 * @generated
	 */
	public static final int FAMILY_MEMBER__DAUGHTERS_INVERSE = 4;

	/**
	 * @generated
	 */
	public static final int FAMILY_MEMBER_FEATURE_COUNT = 5;

	/**
	 * @generated
	 */
	public static FamiliesPackageImpl init() {
		return theFamiliesPackage;
	}

	/**
	 * @generated
	 */
	private static FamiliesPackageImpl theFamiliesPackage;

	/**
	 * @generated
	 */
	protected FamiliesPackageImpl() {
		super();
	}

	/**
	 * @generated
	 */
	@Override
	public EClass getFamilyRegister() {
		return Literals.FAMILY_REGISTER;
	}

	/**
	 * @generated
	 */
	@Override
	public EReference getFamilyRegister_Families() {
		return Literals.FAMILY_REGISTER__FAMILIES;
	}

	/**
	 * @generated
	 */
	@Override
	public EClass getFamily() {
		return Literals.FAMILY;
	}

	/**
	 * @generated
	 */
	@Override
	public EReference getFamily_Father() {
		return Literals.FAMILY__FATHER;
	}

	/**
	 * @generated
	 */
	@Override
	public EReference getFamily_Mother() {
		return Literals.FAMILY__MOTHER;
	}

	/**
	 * @generated
	 */
	@Override
	public EReference getFamily_Sons() {
		return Literals.FAMILY__SONS;
	}

	/**
	 * @generated
	 */
	@Override
	public EReference getFamily_Daughters() {
		return Literals.FAMILY__DAUGHTERS;
	}

	/**
	 * @generated
	 */
	@Override
	public EAttribute getFamily_Name() {
		return Literals.FAMILY__NAME;
	}

	/**
	 * @generated
	 */
	@Override
	public EReference getFamily_FamiliesInverse() {
		return Literals.FAMILY__FAMILIES_INVERSE;
	}

	/**
	 * @generated
	 */
	@Override
	public EClass getFamilyMember() {
		return Literals.FAMILY_MEMBER;
	}

	/**
	 * @generated
	 */
	@Override
	public EAttribute getFamilyMember_Name() {
		return Literals.FAMILY_MEMBER__NAME;
	}

	/**
	 * @generated
	 */
	@Override
	public EReference getFamilyMember_FatherInverse() {
		return Literals.FAMILY_MEMBER__FATHER_INVERSE;
	}

	/**
	 * @generated
	 */
	@Override
	public EReference getFamilyMember_MotherInverse() {
		return Literals.FAMILY_MEMBER__MOTHER_INVERSE;
	}

	/**
	 * @generated
	 */
	@Override
	public EReference getFamilyMember_SonsInverse() {
		return Literals.FAMILY_MEMBER__SONS_INVERSE;
	}

	/**
	 * @generated
	 */
	@Override
	public EReference getFamilyMember_DaughtersInverse() {
		return Literals.FAMILY_MEMBER__DAUGHTERS_INVERSE;
	}

	/**
	 * @generated
	 */
	@Override
	public EFactory getEFactoryInstance() {
		return FamiliesFactoryImpl.init();
	}

	/**
	 * @generated
	 */
	protected EClass familyRegisterEClass = null;

	/**
	 * @generated
	 */
	protected EReference familyRegister__FamiliesEClass = null;

	/**
	 * @generated
	 */
	protected EClass familyEClass = null;

	/**
	 * @generated
	 */
	protected EReference family__FatherEClass = null;

	/**
	 * @generated
	 */
	protected EReference family__MotherEClass = null;

	/**
	 * @generated
	 */
	protected EReference family__SonsEClass = null;

	/**
	 * @generated
	 */
	protected EReference family__DaughtersEClass = null;

	/**
	 * @generated
	 */
	protected EAttribute family__NameEClass = null;

	/**
	 * @generated
	 */
	protected EReference family__FamiliesInverseEClass = null;

	/**
	 * @generated
	 */
	protected EClass familyMemberEClass = null;

	/**
	 * @generated
	 */
	protected EAttribute familyMember__NameEClass = null;

	/**
	 * @generated
	 */
	protected EReference familyMember__FatherInverseEClass = null;

	/**
	 * @generated
	 */
	protected EReference familyMember__MotherInverseEClass = null;

	/**
	 * @generated
	 */
	protected EReference familyMember__SonsInverseEClass = null;

	/**
	 * @generated
	 */
	protected EReference familyMember__DaughtersInverseEClass = null;

	/**
	 * @generated
	 */
	@Override
	public void createPackageContents() {
		// Initialize package
		setnsURI("http://families");
		setnsPrefix("Families");
		
		// Create FamilyRegister class
		familyRegisterEClass = createEClass("FamilyRegister", false);
		familyRegister__FamiliesEClass = createEReference("FamilyRegister", "Families", familyRegisterEClass, FAMILY_REGISTER__FAMILIES, 0, -1, Family.class, false);
		
		// Create Family class
		familyEClass = createEClass("Family", false);
		family__FatherEClass = createEReference("Family", "Father", familyEClass, FAMILY__FATHER, 0, 1, FamilyMember.class, false);
		family__MotherEClass = createEReference("Family", "Mother", familyEClass, FAMILY__MOTHER, 0, 1, FamilyMember.class, false);
		family__SonsEClass = createEReference("Family", "Sons", familyEClass, FAMILY__SONS, 0, -1, FamilyMember.class, false);
		family__DaughtersEClass = createEReference("Family", "Daughters", familyEClass, FAMILY__DAUGHTERS, 0, -1, FamilyMember.class, false);
		family__NameEClass = createEAttribute("Family", "Name", familyEClass, FAMILY__NAME, 0, 1, String.class, false);
		family__FamiliesInverseEClass = createEReference("Family", "FamiliesInverse", familyEClass, FAMILY__FAMILIES_INVERSE, 0, 1, FamilyRegister.class, false);
		
		// Create FamilyMember class
		familyMemberEClass = createEClass("FamilyMember", false);
		familyMember__NameEClass = createEAttribute("FamilyMember", "Name", familyMemberEClass, FAMILY_MEMBER__NAME, 0, 1, String.class, false);
		familyMember__FatherInverseEClass = createEReference("FamilyMember", "FatherInverse", familyMemberEClass, FAMILY_MEMBER__FATHER_INVERSE, 0, 1, Family.class, false);
		familyMember__MotherInverseEClass = createEReference("FamilyMember", "MotherInverse", familyMemberEClass, FAMILY_MEMBER__MOTHER_INVERSE, 0, 1, Family.class, false);
		familyMember__SonsInverseEClass = createEReference("FamilyMember", "SonsInverse", familyMemberEClass, FAMILY_MEMBER__SONS_INVERSE, 0, 1, Family.class, false);
		familyMember__DaughtersInverseEClass = createEReference("FamilyMember", "DaughtersInverse", familyMemberEClass, FAMILY_MEMBER__DAUGHTERS_INVERSE, 0, 1, Family.class, false);
	}

	/**
	 * @generated
	 */
	public EClass createEClass(String name, boolean abstract_) {
		EClass eClass = new MinimalEObjectImpl.Container() {};
		eClass.setClassifierID(0);
		eClass.setName(name);
		eClass.setAbstract(abstract_);
		eClass.setInstanceClassName("Families." + name);
		return eClass;
	}

	/**
	 * @generated
	 */
	public EAttribute createEAttribute(String className, String attrName, EClass eClass, int featureID, int lowerBound, int upperBound, Class<?> instanceClass, boolean unique) {
		EAttribute eAttribute = super.createEAttribute(eClass, featureID);
		eAttribute.setName(attrName);
		eAttribute.setEType(instanceClass);
		eAttribute.setLowerBound(lowerBound);
		eAttribute.setUpperBound(upperBound);
		eAttribute.setUnique(unique);
		return eAttribute;
	}

	/**
	 * @generated
	 */
	public EReference createEReference(String className, String refName, EClass eClass, int featureID, int lowerBound, int upperBound, Class<?> instanceClass, boolean containment) {
		EReference eReference = super.createEReference(eClass, featureID);
		eReference.setName(refName);
		eReference.setEType(instanceClass);
		eReference.setLowerBound(lowerBound);
		eReference.setUpperBound(upperBound);
		eReference.setContainment(containment);
		return eReference;
	}

	/**
	 * @generated
	 */
	public static final class Literals {
		public static final EClass FAMILY_REGISTER = theFamiliesPackage != null ? theFamiliesPackage.familyRegisterEClass : (theFamiliesPackage.familyRegisterEClass = new EClassImpl());
		public static final EReference FAMILY_REGISTER__FAMILIES = theFamiliesPackage != null ? theFamiliesPackage.familyRegister__FamiliesEClass : (theFamiliesPackage.familyRegister__FamiliesEClass = new EReferenceImpl());
		public static final EClass FAMILY = theFamiliesPackage != null ? theFamiliesPackage.familyEClass : (theFamiliesPackage.familyEClass = new EClassImpl());
		public static final EReference FAMILY__FATHER = theFamiliesPackage != null ? theFamiliesPackage.family__FatherEClass : (theFamiliesPackage.family__FatherEClass = new EReferenceImpl());
		public static final EReference FAMILY__MOTHER = theFamiliesPackage != null ? theFamiliesPackage.family__MotherEClass : (theFamiliesPackage.family__MotherEClass = new EReferenceImpl());
		public static final EReference FAMILY__SONS = theFamiliesPackage != null ? theFamiliesPackage.family__SonsEClass : (theFamiliesPackage.family__SonsEClass = new EReferenceImpl());
		public static final EReference FAMILY__DAUGHTERS = theFamiliesPackage != null ? theFamiliesPackage.family__DaughtersEClass : (theFamiliesPackage.family__DaughtersEClass = new EReferenceImpl());
		public static final EAttribute FAMILY__NAME = theFamiliesPackage != null ? theFamiliesPackage.family__NameEClass : (theFamiliesPackage.family__NameEClass = new EAttributeImpl());
		public static final EReference FAMILY__FAMILIES_INVERSE = theFamiliesPackage != null ? theFamiliesPackage.family__FamiliesInverseEClass : (theFamiliesPackage.family__FamiliesInverseEClass = new EReferenceImpl());
		public static final EClass FAMILY_MEMBER = theFamiliesPackage != null ? theFamiliesPackage.familyMemberEClass : (theFamiliesPackage.familyMemberEClass = new EClassImpl());
		public static final EAttribute FAMILY_MEMBER__NAME = theFamiliesPackage != null ? theFamiliesPackage.familyMember__NameEClass : (theFamiliesPackage.familyMember__NameEClass = new EAttributeImpl());
		public static final EReference FAMILY_MEMBER__FATHER_INVERSE = theFamiliesPackage != null ? theFamiliesPackage.familyMember__FatherInverseEClass : (theFamiliesPackage.familyMember__FatherInverseEClass = new EReferenceImpl());
		public static final EReference FAMILY_MEMBER__MOTHER_INVERSE = theFamiliesPackage != null ? theFamiliesPackage.familyMember__MotherInverseEClass : (theFamiliesPackage.familyMember__MotherInverseEClass = new EReferenceImpl());
		public static final EReference FAMILY_MEMBER__SONS_INVERSE = theFamiliesPackage != null ? theFamiliesPackage.familyMember__SonsInverseEClass : (theFamiliesPackage.familyMember__SonsInverseEClass = new EReferenceImpl());
		public static final EReference FAMILY_MEMBER__DAUGHTERS_INVERSE = theFamiliesPackage != null ? theFamiliesPackage.familyMember__DaughtersInverseEClass : (theFamiliesPackage.familyMember__DaughtersInverseEClass = new EReferenceImpl());
	}

	/**
	 * @generated
	 */
	private static class EClassImpl extends MinimalEObjectImpl implements EClass {
	}

	/**
	 * @generated
	 */
	private static class EAttributeImpl extends MinimalEObjectImpl implements EAttribute {
	}

	/**
	 * @generated
	 */
	private static class EReferenceImpl extends MinimalEObjectImpl implements EReference {
	}
} // FamiliesPackageImpl