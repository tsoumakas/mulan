/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package mulan.classifier.meta;

import java.util.Set;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.HierarchyBuilder.Method;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.DataUtils;
import mulan.data.LabelsMetaData;
import mulan.data.MultiLabelInstances;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 <!-- globalinfo-start -->
 * Class implementing the Hierarchy Of Multi-labEl leaRners algorithm. For more information, see<br>
 * <br>
 * Grigorios Tsoumakas, Ioannis Katakis, Ioannis Vlahavas: Effective and Efficient Multilabel Classification in Domains with Large Number of Labels. In: Proc. ECML/PKDD 2008 Workshop on Mining Multidimensional Data (MMD'08), 2008.
 * <br>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{Tsoumakas2008,
 *    author = {Grigorios Tsoumakas and Ioannis Katakis and Ioannis Vlahavas},
 *    booktitle = {Proc. ECML/PKDD 2008 Workshop on Mining Multidimensional Data (MMD'08)},
 *    title = {Effective and Efficient Multilabel Classification in Domains with Large Number of Labels},
 *    year = {2008},
 *    location = {Antwerp, Belgium}
 * }
 * </pre>
 * <br>
 <!-- technical-bibtex-end -->
 *
 * @author Grigorios Tsoumakas
 * @version 2012.02.27
 */
public class HOMER extends MultiLabelMetaLearner {

    private final int numClusters;
    private HMC hmc;
    private HierarchyBuilder hb;
    private Instances header;
    private Method method;
    private MultiLabelInstances m;
    private int numMetaLabels;

    /**
     * Default constructor
     */
    public HOMER() {
        super(new BinaryRelevance(new J48()));
        method = HierarchyBuilder.Method.BalancedClustering;
        numClusters = 3;
    }

    /**
     * Creates a new instance based on given multi-label learner, number of 
     * children and partitioning method
     * 
     * @param mll multi-label learner
     * @param clusters number of partitions
     * @param method partitioning method
     */
    public HOMER(MultiLabelLearner mll, int clusters, Method method) {
        super(mll);
        this.method = method;
        numClusters = clusters;
    }

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        debug("Learning the hierarchy of models");
        hb = new HierarchyBuilder(numClusters, method);
        LabelsMetaData labelHierarchy = hb.buildLabelHierarchy(trainingSet);

        debug("Constructing the hierarchical multilabel dataset");
        MultiLabelInstances meta = HierarchyBuilder.createHierarchicalDataset(trainingSet, labelHierarchy);
        header = new Instances(meta.getDataSet(), 0);

        debug("Training the hierarchical classifier");
        hmc = new HMC(baseLearner);
        hmc.setDebug(getDebug());
        hmc.build(meta);

        Set<String> leafLabels = trainingSet.getLabelsMetaData().getLabelNames();
        Set<String> metaLabels = labelHierarchy.getLabelNames();
        for (String string : leafLabels) {
            metaLabels.remove(string);
        }
        numMetaLabels = metaLabels.size();
    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        Instance transformed = DataUtils.createInstance(instance, instance.weight(), instance.toDoubleArray());
        for (int i = 0; i < numMetaLabels; i++) {
            transformed.insertAttributeAt(transformed.numAttributes());
        }

        transformed.setDataset(header);
        MultiLabelOutput mlo = hmc.makePrediction(transformed);
        boolean[] oldBipartition = mlo.getBipartition();
        //System.out.println("old:" + Arrays.toString(oldBipartition));
        boolean[] newBipartition = new boolean[numLabels];
        System.arraycopy(oldBipartition, 0, newBipartition, 0, numLabels);
        //System.out.println("new:" + Arrays.toString(newBipartition));
        double[] oldConfidences = mlo.getConfidences();
        double[] newConfidences = new double[numLabels];
        System.arraycopy(oldConfidences, 0, newConfidences, 0, numLabels);
        MultiLabelOutput newMLO = new MultiLabelOutput(newBipartition, newConfidences);
        return newMLO;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Grigorios Tsoumakas and Ioannis Katakis and Ioannis Vlahavas");
        result.setValue(Field.TITLE, "Effective and Efficient Multilabel Classification in Domains with Large Number of Labels");
        result.setValue(Field.BOOKTITLE, "Proc. ECML/PKDD 2008 Workshop on Mining Multidimensional Data (MMD'08)");
        result.setValue(Field.LOCATION, "Antwerp, Belgium");
        result.setValue(Field.YEAR, "2008");
        return result;
    }

    //spark temporary edit for complexity measures   
    
    
    /**
     * Returns the number of nodes
     * 
     * @return number of nodes
     */
    public long getNoNodes() {
        return hmc.getNoNodes();
    }

    /**
     * Returns the number of classifier evaluations
     * 
     * @return number of classifier evaluations
     */
    public long getNoClassifierEvals() {
        return hmc.getNoClassifierEvals();
    }

    /**
     * Returns the total number of instances used for training
     * 
     * @return total number of instances used for training
     */
    public long getTotalUsedTrainInsts() {
        return hmc.getTotalUsedTrainInsts();
    }

    public String globalInfo() {
        return "Class implementing the Hierarchy Of Multi-labEl leaRners " +
               "algorithm. For more information, see\n\n"
                + getTechnicalInformation().toString();
    }
}