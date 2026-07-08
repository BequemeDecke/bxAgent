public abstract class BXToolAdapterFactory<S, T, D, A> {
    protected String name;

    public BXToolAdapterFactory(String transformationName) {
        this.name = transformationName;
    }

    public abstract A createAgentDecision(Configurator<D> conf);
    public abstract AgentTransformationForEMF<S, T, A> createTransformation();

    public TransformationBxToolAdapter<S, T, D, A> createBxToolAdapter(BiConsumer<S,S> source, BiConsumer<T,T> target) {
        var transformation = createTransformation();
        var adapter = new TransformationBxToolAdapter(source, target, this::createAgentDecision, transformation, this.name);
        return adapter;
    }
}