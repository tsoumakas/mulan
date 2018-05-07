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

import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import mulan.transformations.RemoveAllLabels;
import weka.classifiers.trees.J48;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.*;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 <!-- globalinfo-start -->
 * Class implementing clustering-based multi-label classification. For more information, see<br>
 * <br>
 * Gulisong Nasierding, Grigorios Tsoumakas, Abbas Kouzani: Clustering Based Multi-Label Classification for Image Annotation and Retrieval. In: Proc. 2009 IEEE International Conference on Systems, Man, and Cybernetics (SMC 2009), 2009.
 * <br>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{GulisongNasierding2009,
 *    author = {Gulisong Nasierding, Grigorios Tsoumakas, Abbas Kouzani},
 *    booktitle = {Proc. 2009 IEEE International Conference on Systems, Man, and Cybernetics (SMC 2009)},
 *    title = {Clustering Based Multi-Label Classification for Image Annotation and Retrieval},
 *    year = {2009},
 *    location = {Texas, USA}
 * }
 * </pre>
 * <br>
 <!-- technical-bibtex-end -->
 *
 * @author  Gulisong Nasierding
 * @author  Grigorios Tsoumakas
 * @version 2012.02.27
 */
public class ClusteringBased extends MultiLabelMetaLearner {

    /** The number of clusters */
    private int numClusters;
    /** The multi-label learners, one for each cluster */
    private MultiLabelLearner[] multi;
    /** The clusterer to use */
    private Clusterer clusterer;

    /**
     * Default constructor. 
     */
    public ClusteringBased() {
        super(new LabelPowerset(new J48()));
        try {
            SimpleKMeans kmeans = new SimpleKMeans();
            kmeans.setNumClusters(5);
            kmeans.setDistanceFunction(new EuclideanDistance());
            clusterer = kmeans;
        } catch (Exception ex) {
            Logger.getLogger(ClusteringBased.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    /**
     * Constructor
     * 
     * @param aClusterer the clustering approach
     * @param aMultiLabelClassifier the multi-label learner
     */
    public ClusteringBased(Clusterer aClusterer, MultiLabelLearner aMultiLabelClassifier) {
        super(aMultiLabelClassifier);
        clusterer = aClusterer;
    }

    /**
     * Returns the clustering approach
     * 
     * @return the clustering approach
     */
    public Clusterer getClusterer() {
        return clusterer;
    }

    @Override
    public void buildInternal(MultiLabelInstances trainData) throws Exception {
        Instances trainInstances = trainData.getDataSet();

        // remove label attributes and cluster data
        Instances removedInstances = RemoveAllLabels.transformInstances(trainData);
        clusterer.buildClusterer(removedInstances);
        numClusters = clusterer.numberOfClusters();

        // initialize multi-label datasets for each cluster
        MultiLabelInstances[] subsetMultiLabelInstances = new MultiLabelInstances[numClusters];
        Instances[] subsetInstances = new Instances[numClusters];

        for (int i = 0; i < numClusters; i++) {
            subsetInstances[i] = new Instances(trainInstances, 0);
            subsetMultiLabelInstances[i] = new MultiLabelInstances(subsetInstances[i], trainData.getLabelsMetaData());
        }
        // partition data according to cluster
        for (int i = 0; i < trainInstances.numInstances(); i++) {
            int clusterOfInstance = clusterer.clusterInstance(removedInstances.instance(i));
            subsetMultiLabelInstances[clusterOfInstance].getDataSet().add(trainInstances.instance(i));
        }

        // build a multi-label classifier from each subset
        multi = new MultiLabelLearner[numClusters];
        for (int i = 0; i < numClusters; i++) {
            try {
                multi[i] = baseLearner.makeCopy();
                debug("Dataset " + (i+1) + ": " + subsetMultiLabelInstances[i].getDataSet().numInstances() + " instances");
                multi[i].build(subsetMultiLabelInstances[i]);


            } catch (Exception ex) {
                Logger.getLogger(ClusteringBased.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {
        Instance newInstance = RemoveAllLabels.transformInstance(instance, labelIndices);
        int cluster = clusterer.clusterInstance(newInstance);
        return multi[cluster].makePrediction(instance);
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Gulisong Nasierding, Grigorios Tsoumakas, Abbas Kouzani");
        result.setValue(Field.TITLE, "Clustering Based Multi-Label Classification for Image Annotation and Retrieval");
        result.setValue(Field.BOOKTITLE, "Proc. 2009 IEEE International Conference on Systems, Man, and Cybernetics (SMC 2009)");
        result.setValue(Field.LOCATION, "Texas, USA");
        result.setValue(Field.YEAR, "2009");
        return result;
    }
    
    public String globalInfo() {
        return "Class implementing clustering-based multi-label classification."
                + " For more information, see\n\n" 
                + getTechnicalInformation().toString(); 
    }
}