package weka.core.neighboursearch;



import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

/**
Override the function of kNearestNeighbours by get the indices of k nearest neighbours of the target instance

@author Bin Liu
@version 2019.3.21
 */
public class LinearNNSearch2 extends LinearNNSearch {

  /** for serialization. */
  private static final long serialVersionUID = 1915484723703917242L;

 
  /**
   * Constructor. Needs setInstances(Instances) 
   * to be called before the class is usable.
   */
  public LinearNNSearch2() {
    super();
  }
  
  /**
   * Constructor that uses the supplied set of 
   * instances.
   * 
   * @param insts	the instances to use
   */
  public LinearNNSearch2(Instances insts) {
    super(insts);
    m_DistanceFunction.setInstances(insts);
  }

  
  /**
   * 
   * Returns k nearest instances of the current neighbourhood from supplied
   * instance.
   *  
   * @param target 	The instance to find the k nearest neighbours for.
   * @param kNN		The number of nearest neighbours to find.
   * @param indices To save the indices of k nearest neighbours in target.
   * @return		the k nearest neighbours
   * @throws Exception  if the neighbours could not be found.
   */
  public KnnResult kNearestNeighbours2(Instance target, int kNN) throws Exception {
  
    //debug
    boolean print=false;

    if(m_Stats!=null)
      m_Stats.searchStart();
 
    MyHeap heap = new MyHeap(kNN);
    //System.out.println(m_Instances.numInstances());
    double distance; int firstkNN=0;
    for(int i=0; i<m_Instances.numInstances(); i++) {
      if(target == m_Instances.instance(i)) //for hold-one-out cross-validation
        continue;
      if(m_Stats!=null) 
        m_Stats.incrPointCount();
      if(firstkNN<kNN) {
        if(print)
          System.out.println("K(a): "+(heap.size()+heap.noOfKthNearest()));
        distance = m_DistanceFunction.distance(target, m_Instances.instance(i), Double.POSITIVE_INFINITY, m_Stats);
        if(distance == 0.0 && m_SkipIdentical)
          if(i<m_Instances.numInstances()-1)
            continue;
          else
            heap.put(i, distance);
        heap.put(i, distance);
        firstkNN++;
      }
      else {
        MyHeapElement temp = heap.peek();
        if(print)
          System.out.println("K(b): "+(heap.size()+heap.noOfKthNearest()));
        distance = m_DistanceFunction.distance(target, m_Instances.instance(i), temp.distance, m_Stats);
        if(distance == 0.0 && m_SkipIdentical)
          continue;
        if(distance < temp.distance) {
          heap.putBySubstitute(i, distance);
        }
        else if(distance == temp.distance) {
          heap.putKthNearest(i, distance);
        }

      }
    }
    
    Instances neighbours = new Instances(m_Instances, (heap.size()+heap.noOfKthNearest()));
    m_Distances = new double[heap.size()+heap.noOfKthNearest()];
    int []indices = new int[heap.size()+heap.noOfKthNearest()];
    int i=1; MyHeapElement h;
    while(heap.noOfKthNearest()>0) {
      h = heap.getKthNearest();
      indices[indices.length-i] = h.index;
      m_Distances[indices.length-i] = h.distance;
      i++;
    }
    while(heap.size()>0) {
      h = heap.get();
      indices[indices.length-i] = h.index;
      m_Distances[indices.length-i] = h.distance;
      i++;
    }
    
    m_DistanceFunction.postProcessDistances(m_Distances);
    
    for(int k=0; k<indices.length; k++) {
      neighbours.add(m_Instances.instance(indices[k]));
    }
    
    if(m_Stats!=null)
      m_Stats.searchFinish();
    
    
    return new KnnResult(neighbours, indices);    
  }
  
}


