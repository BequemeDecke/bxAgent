package com.example.transformation;

import static org.junit.Assert.fail;

import java.io.IOException;
import java.util.function.Supplier;

import org.benchmarx.config.Configurator;
import org.benchmarx.edit.IEdit;
import org.benchmarx.config.Configurator;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.util.EcoreUtil;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

import com.example.transformation.Decisions;
import com.example.source.Comparator;
import com.example.target.Comparator;
import com.example.source.Factory;
import com.example.source.Register;
import com.example.target.Factory;
import com.example.target.Register;
import com.example.transformation.TransformationImplementation;

import com.example.additional.Import1;
import com.example.additional.Import2;


public class TransformationImplementation extends BXToolForEMF<SourceRegister, TargetRegister, Decisions> {
	private ResourceSet set = new ResourceSetImpl();
	private Resource source;
	private Resource target;
	private Resource corr;
	private TransformationImplementation javaTrans;
	private Configurator<Decisions> conf;
	private Configurator<Decisions> defaultConf;
	private SourceRegister sourceRegisterInstance;
	private TargetRegister targetRegisterInstance;

	private static final String RESULTPATH = "results/Java";

	public TransformationImplementation() {
		super(new SourceComparator(), new TargetComparator());
	}
	
	@Override
	public String getName() {
		return "TransformationImplementation";
	}

	@Override
	public void initiateSynchronisationDialogue() {
		//setConfigurator(new Configurator<Decisions>().makeDecision(Decisions.PREFER_CREATING_PARENT_TO_CHILD, true)
		//	  	.makeDecision(Decisions.PREFER_EXISTING_FAMILY_TO_NEW, true));
		setConfigurator();

		set.getResourceFactoryRegistry().getExtensionToFactoryMap().put("SourceModel", new XMIResourceFactoryImpl());
		set.getResourceFactoryRegistry().getExtensionToFactoryMap().put("TargetModel", new XMIResourceFactoryImpl());
		set.getResourceFactoryRegistry().getExtensionToFactoryMap().put("corr", new XMIResourceFactoryImpl());

		source = set.createResource(URI.createURI("sourceModel.SourceModel"));
		target = set.createResource(URI.createURI("targetModel.TargetModel"));
		corr = set.createResource(URI.createURI("corrModel.corr"));
        
		sourceRegisterInstance = SourceFactory.eINSTANCE.createSourceRegister();
		targetRegisterInstance = TargetFactory.eINSTANCE.createTargetRegister();
		source.getContents().add(sourceRegisterInstance);
		target.getContents().add(targetRegisterInstance);

		initiateDialogue();
	}
	
	/**
	 * Perform an edit delta on the target model and propagate the change to the
	 * source model
	 * 
	 * @param edit : the source edit delta
	 */
	@Override
	public void performAndPropagateTargetEdit(Supplier<IEdit<TargetRegister>> edit) {
		edit.get();
		performAndPropagateTargetEdit();
	}

	/**
	 * Perform an edit delta on the source model and propagate the change to the
	 * target model
	 * 
	 * @param edit : the source edit delta
	 */
	@Override
	public void performAndPropagateSourceEdit(Supplier<IEdit<SourceRegister>> edit) {
		edit.get();		
		performAndPropagateSourceEdit();
	}	

	@Override
	public void performAndPropagateEdit(Supplier<IEdit<SourceRegister>> sourceEdit,
			Supplier<IEdit<TargetRegister>> targetEdit) {
		// TODO Auto-generated method stub
		//fail("Concurrent edits not supported.");
		sourceEdit.get();
		targetEdit.get();
		performAndPropagateConcurrentEdit();
	}

	@Override
	public void setConfigurator(Configurator<Decisions> configurator) {
		if (defaultConf == null)
			defaultConf = configurator;
		conf = configurator;
	}

	@Override
	public SourceRegister getSourceModel() {
		return sourceRegisterInstance;
	}

	@Override
	public TargetRegister getTargetModel() {
		return targetRegisterInstance;
	}

	@Override
	public void saveModels(String name) {
		ResourceSet set = new ResourceSetImpl();
		set.getResourceFactoryRegistry().getExtensionToFactoryMap().put(Resource.Factory.Registry.DEFAULT_EXTENSION,
				new XMIResourceFactoryImpl());
		URI srcURI = URI.createFileURI(RESULTPATH + "/" + name + "SourceModel.xmi");
		URI trgURI = URI.createFileURI(RESULTPATH + "/" + name + "TargetModel.xmi");
		Resource resSource = set.createResource(srcURI);
		Resource resTarget = set.createResource(trgURI);

		EObject colSource = EcoreUtil.copy(getSourceModel());
		EObject colTarget = EcoreUtil.copy(getTargetModel());

		resSource.getContents().add(colSource);
		resTarget.getContents().add(colTarget);

		try {
			resSource.save(null);
			resTarget.save(null);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
