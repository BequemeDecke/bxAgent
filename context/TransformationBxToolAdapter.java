import java.util.function.BiConsumer;
import java.util.function.Supplier;
import java.util.function.Function;

import org.benchmarx.config.Configurator;
import org.benchmarx.edit.IEdit;
import org.benchmarx.emf.BXToolForEMF;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.util.EcoreUtil;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

public class TransformationBxToolAdapter<S, T, D, A> extends BXToolForEMF<S, T, D> {
    private ResourceSet set = new ResourceSetImpl();
	private Resource source;
	private Resource target;
	private Resource corr;

	private AgentTransformationForEMF<S, T, A> transformation;
    private Function<Configurator<D>, A> configMapper;
    private String name;

	private Configurator<D> conf;
	private Configurator<D> defaultConf;
	private S sourceRoot;
	private T targetRoot;

	private static final String RESULTPATH = "results/Java";

	public TransformationBxToolAdapter(BiConsumer<S,S> source, BiConsumer<T,T> target, Function<Configurator<D>, A> configMapper, AgentTransformation<S, T, A> transformation, String name) {
		super(src, trg);

        this.configMapper = configMapper;
        this.transformation = transformation;
        this.name = name;
	}
	
	@Override
	public String getName() {
		return this.name;
	}

	@Override
	public void initiateSynchronisationDialogue() {
        // TODO: Big Problema, wo werden Decisions registriert?
		setConfigurator(new Configurator<D>().makeDecision(Decisions.PREFER_CREATING_PARENT_TO_CHILD, true)
				.makeDecision(Decisions.PREFER_EXISTING_FAMILY_TO_NEW, true));

		set.getResourceFactoryRegistry().getExtensionToFactoryMap().put("family", new XMIResourceFactoryImpl());
		set.getResourceFactoryRegistry().getExtensionToFactoryMap().put("person", new XMIResourceFactoryImpl());
		set.getResourceFactoryRegistry().getExtensionToFactoryMap().put("corr", new XMIResourceFactoryImpl());

		source = set.createResource(URI.createURI("sourceModel.family"));
		target = set.createResource(URI.createURI("targetModel.person"));
		corr = set.createResource(URI.createURI("corrModel.corr"));
		sourceRoot = FamiliesFactory.eINSTANCE.createFamilyRegister();
		targetRoot = PersonsFactory.eINSTANCE.createPersonRegister();
		source.getContents().add(sourceRoot);
		target.getContents().add(targetRoot);
        transformation.setSource(sourceRoot);
        transformation.setTarget(targetRoot);
		transformation.transformSourceToTarget(sourceRoot, targetRoot);		
	}
	
	/**
	 * Perform an edit delta on the target model and propagate the change to the
	 * source model
	 * 
	 * @param edit : the source edit delta
	 */
	@Override
	public void performAndPropagateTargetEdit(Supplier<IEdit<PersonRegister>> edit) {
		edit.get();
		transformation.backward(targetRoot, sourceRoot, 
				conf.decide(Decisions.PREFER_EXISTING_FAMILY_TO_NEW), conf.decide(Decisions.PREFER_CREATING_PARENT_TO_CHILD));
	}

	/**
	 * Perform an edit delta on the source model and propagate the change to the
	 * target model
	 * 
	 * @param edit : the source edit delta
	 */
	@Override
	public void performAndPropagateSourceEdit(Supplier<IEdit<FamilyRegister>> edit) {
		edit.get();		
		transformation.forward(sourceRoot, targetRoot);
	}	

	@Override
	public void performAndPropagateEdit(Supplier<IEdit<FamilyRegister>> sourceEdit,
			Supplier<IEdit<PersonRegister>> targetEdit) {
		// TODO Auto-generated method stub
		//fail("Concurrent edits not supported.");
		sourceEdit.get();
		targetEdit.get();
		transformation.synch(sourceRoot, targetRoot, 
				conf.decide(Decisions.PREFER_EXISTING_FAMILY_TO_NEW), conf.decide(Decisions.PREFER_CREATING_PARENT_TO_CHILD));
	}

	@Override
	public void setConfigurator(Configurator<Decisions> configurator) {
		if (defaultConf == null)
			defaultConf = configurator;
		conf = configurator;
	}

	@Override
	public FamilyRegister getSourceModel() {
		return sourceRoot;
	}

	@Override
	public PersonRegister getTargetModel() {
		return targetRoot;
	}

	@Override
	public void saveModels(String name) {
		ResourceSet set = new ResourceSetImpl();
		set.getResourceFactoryRegistry().getExtensionToFactoryMap().put(Resource.Factory.Registry.DEFAULT_EXTENSION,
				new XMIResourceFactoryImpl());
		URI srcURI = URI.createFileURI(RESULTPATH + "/" + name + "Family.xmi");
		URI trgURI = URI.createFileURI(RESULTPATH + "/" + name + "Person.xmi");
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