public interface AgentTransformationForEMF<S, T, A> {
    void forward(source: S, target: T, decisions: A);
    void backward(target: T, source: S, decisions: A);
    void synch(source: S, target: T);

    void transformSourceToTarget(source: S, target: T, decisions: A);
    void transformTargetToSource(target: T, source: S, decisions: A);
}