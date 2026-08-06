# Distributed Systems

## Overview

A distributed system consists of multiple computing processes that communicate over a network to provide a shared capability. The processes may run on different machines, containers, or virtual machines and may fail independently.

Distributed systems are useful because they allow applications to scale beyond the resources of a single machine and can improve availability by distributing workloads across multiple instances.

They also introduce challenges that do not exist in the same form inside a single process. Network communication is slower and less reliable than local function calls, machines can fail independently, and different components may observe different states at different times.

A distributed system therefore needs explicit strategies for communication, consistency, failure handling, resource management, and observability.

## Communication Between Services

Distributed components commonly communicate using HTTP APIs, remote procedure calls, message queues, or event-streaming systems.

An HTTP request between services may look similar to a local function call from the perspective of application code, but the behavior is fundamentally different. The remote service can be unavailable, slow, overloaded, or unreachable.

Network requests should therefore have explicit timeouts. Without a timeout, a waiting request can consume resources indefinitely and eventually cause a chain of failures in dependent services.

Retries can help recover from transient failures. However, retries also increase traffic and can make an overloaded service even less stable. Retry policies should therefore use bounded attempts and appropriate delays.

Idempotency is particularly important when retrying operations. An operation that can safely be performed more than once is easier to retry than one that creates irreversible side effects each time it executes.

## Synchronous and Asynchronous Communication

Synchronous communication requires the caller to wait for a response. It is appropriate when the caller needs an immediate result and the expected latency is manageable.

Asynchronous communication separates the sender from the receiver. A message can be placed onto a queue and processed later by another component.

Message queues can smooth traffic spikes because producers and consumers do not need to operate at exactly the same rate. A consumer can process messages as capacity becomes available.

Queues also introduce new considerations. Messages may be delayed, delivered more than once, or remain unprocessed when consumers fail.

Consumers should therefore be designed with appropriate error handling and, where possible, idempotent processing. Monitoring queue depth and processing latency can also reveal when the system is falling behind.

The choice between synchronous and asynchronous communication depends on application requirements. A user-facing prediction request may require a synchronous response, while a large batch-processing workflow may benefit from asynchronous execution.

## Consistency

In a single-process application, state changes can often be observed through a shared memory model. Distributed systems do not have this luxury because each process maintains its own local view of state.

Strong consistency attempts to ensure that reads observe the latest committed state according to the system's consistency model. Eventual consistency allows replicas to temporarily disagree while converging over time.

Neither approach is universally correct. Strong consistency can require additional coordination and increase latency, while eventual consistency can simplify some scalable architectures at the cost of temporarily stale data.

Database replication is a common example. A primary database may accept writes while one or more replicas serve reads. If replication is asynchronous, a recently committed change may not immediately appear on a replica.

Applications must therefore understand the consistency guarantees of their storage and communication systems rather than assuming all copies of data are synchronized immediately.

## Replication and Scaling

Replication creates multiple instances of a service or data store. Replicas can distribute workload and provide additional capacity.

Stateless application services are particularly suitable for horizontal scaling. If request state is stored externally, multiple containers can process requests without requiring clients to return to the same instance.

A load balancer can distribute incoming requests across available service instances. Load balancing strategies may consider simple rotation, current load, or other characteristics.

Scaling application instances does not automatically scale dependencies. Every service instance may maintain database connections, caches, or other resources. Increasing the number of replicas can therefore increase load on shared infrastructure.

Database systems may use read replicas, partitioning, or sharding to increase capacity. These techniques introduce their own consistency and operational considerations.

Capacity planning should consider the complete dependency graph. A service that can process twice as many requests may simply move the bottleneck to the database or another downstream dependency.

## Fault Tolerance

Distributed components fail independently. A database can become unreachable while an API remains operational, or a single service instance can fail while other replicas continue processing requests.

Fault-tolerant systems attempt to continue operating when individual components fail. Redundancy, replication, timeouts, retries, circuit breakers, and graceful degradation are common techniques.

A circuit breaker can temporarily stop sending requests to a failing dependency. This prevents repeated failures from consuming resources and gives the dependency an opportunity to recover.

Graceful degradation means that a system provides a reduced capability when a dependency is unavailable. For example, an application might serve cached information when a non-critical backend service cannot be reached.

Fault tolerance requires understanding which failures are recoverable. Retrying a temporary network timeout may be useful, while retrying an invalid request repeatedly is unlikely to help.

Recovery procedures should also consider partial failures. A request may reach a remote service and cause an operation to complete even if the response is lost. Retrying such an operation without considering idempotency can produce duplicate effects.

## Caching

Caching is frequently used in distributed systems to reduce latency and decrease load on shared services.

A cache can store frequently requested data near the application that needs it. Subsequent requests can avoid network communication with the original data source.

Distributed caches introduce consistency challenges. Multiple application instances may access the same cache, and cached data may become stale relative to the authoritative source.

Cache invalidation can be based on time-to-live values, explicit invalidation events, or versioning. The appropriate strategy depends on how much staleness the application can tolerate.

Caches can also create failure modes. If many cached entries expire simultaneously, a large number of requests may reach the database at once. This can create a sudden load spike known as a cache stampede.

Caching should therefore be treated as part of system architecture rather than simply an optimization added without considering failure behavior.

## Batching and Throughput

Distributed communication often has a fixed cost per request. Sending many small requests can therefore be significantly less efficient than sending fewer larger batches.

Batching groups multiple operations into a single request. This reduces network round trips and can improve throughput.

Batching is particularly useful for machine-learning inference and data-processing workloads. A model-serving service can process multiple inputs in one inference operation, while a database client can submit multiple records in a single transaction.

Larger batches are not always better. They can increase memory usage and waiting time, which may be unacceptable for latency-sensitive workloads.

Systems that support both interactive and batch workloads may need separate processing paths. Interactive requests can prioritize low latency, while background jobs can prioritize throughput.

Performance measurements should therefore consider both throughput and latency. A system that processes more operations per second may still provide an unacceptable user experience if individual requests become too slow.

## Resource Management

Distributed services consume resources at multiple levels. CPU and memory usage affect individual processes, while database connections, network bandwidth, and queue capacity affect the system as a whole.

A service should avoid creating unlimited concurrent operations. Connection pools, worker limits, bounded queues, and concurrency controls can prevent a traffic spike from exhausting resources.

Backpressure is a mechanism by which a system communicates that downstream capacity is limited. A queue can grow when producers are faster than consumers, or a service can reject requests when it reaches a defined capacity limit.

Without backpressure, overload can propagate through the system. A slow database can cause application requests to accumulate, consuming more memory and connections until additional components begin failing.

Capacity limits should therefore be intentional and observable. Metrics can show resource utilization and reveal when a system is approaching its operating limits.

## Observability

Distributed systems are difficult to diagnose using logs from a single component. A user request may travel through an API, database, message queue, model-serving service, and several other components.

Structured logging can provide consistent information such as request identifiers, operation names, durations, and error details.

Metrics provide aggregated measurements such as request rate, latency, error rate, queue depth, and resource consumption. These measurements can reveal trends and help identify bottlenecks.

Distributed tracing connects operations across service boundaries. A trace can show that a slow API request spent most of its time waiting for a database query or a remote model inference call.

Observability should be designed into distributed systems rather than added only after failures occur. The combination of logs, metrics, and traces provides complementary perspectives on system behavior.

Correlation identifiers are particularly useful. A request identifier propagated across services allows operators to connect logs belonging to the same logical operation.

## Deployment and Containerization

Containers are commonly used to deploy distributed services. Each service can be packaged with its runtime dependencies and deployed as one or more container instances.

Container orchestration systems can manage service replicas, networking, health checks, resource limits, and restarts.

A service should not assume that another container is permanently available. Dependencies can restart, network connections can disappear, and deployments can temporarily reduce capacity.

Health checks can distinguish between a process that is running and a service that is ready to accept requests. Readiness is especially important during rolling deployments because traffic should not be sent to an instance before its dependencies and application state are prepared.

Deployment strategies can reduce risk by gradually introducing new versions. If monitoring reveals elevated error rates or latency, traffic can be reduced or returned to the previous version.

## Reliability Engineering

Reliability is a system-level property. Individual components may work correctly while their interaction produces unexpected failures.

Failure scenarios should therefore be considered explicitly. Useful questions include what happens when a database becomes unavailable, when a remote request times out, when a queue grows faster than consumers can process it, or when one service instance repeatedly crashes.

Testing these scenarios can reveal weaknesses that normal functional tests do not expose. Load tests can evaluate capacity, while failure-injection techniques can examine recovery behavior.

Operational metrics should support clear decisions. Latency percentiles, error rates, saturation, queue depth, and dependency availability can indicate whether a service is operating within expected limits.

Reliable distributed systems do not assume that failures can be eliminated. Instead, they make failures detectable, limit their impact, and provide controlled recovery mechanisms.

## Architectural Trade-offs

Distributed architecture is not automatically better than a monolithic design. Splitting an application into services introduces network communication, deployment complexity, monitoring requirements, and additional failure modes.

A distributed design becomes valuable when independent scaling, deployment, organizational boundaries, fault isolation, or workload characteristics justify the added complexity.

Components should have clear responsibilities and well-defined contracts. Dependency injection and explicit interfaces can make service boundaries easier to test and evolve.

The best architecture depends on actual requirements. A small application may be simpler and more reliable as a single process, while a high-volume platform may require multiple independently scalable services.

Distributed systems engineering is ultimately about managing trade-offs. Scalability, latency, consistency, availability, resource usage, and operational complexity must be considered together rather than optimized independently.
