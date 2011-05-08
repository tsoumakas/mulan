package mulan.data;

/**
 *  An interface for various types of dependency identification between pairs of labels.
 * .
 * @author Lena Chekina (lenat@bgu.ac.il)
 * @version 30.11.2010
 */
public interface  LabelPairsDependenceIdentifier {

    /**
     *  Calculates dependence level between each pair of labels in the given multilabel data set
     *
     * @param mlInstances multilabel data set
     * @return an array of label pairs sorted in descending order of pairs' dependence score
     */
    public LabelsPair[] calculateDependence(MultiLabelInstances mlInstances);

    /**
     * Returns a critical value
     * @return critical value
     */
    public double getCriticalValue();

}