public interface AgentTransformationForEMF<S, T, A> {
    void forward(S source, T target, A decisions);
    void backward(T target, S source, A decisions);
    void synch(S source, T target);

    void transformSourceToTarget(S source, T target, A decisions);
    void transformTargetToSource(T target, S source, A decisions);
}