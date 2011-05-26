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

/*
 *    HOMER.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.classifier.meta;

import java.util.Set;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.data.DataUtils;
import mulan.classifier.meta.HierarchyBuilder.Method;
import mulan.data.LabelsMetaData;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.*;

/**
 * <!-- globalinfo-start -->
 *
 * <pre>
 * Class implementing the HOMER algorithm
 * </pre>
 *
 * For more information:
 *
 * <pre>
 * G. Tsoumakas, I. Katakis, I. Vlahavas, "Effective and Efficient Multilabel
 * Classification in Domains with Large Number of Labels", Proc. ECML/PKDD 2008
 * Workshop on Mining Multidimensional Data (MMD'08), Antwerp, Belgium, 2008.
 * </pre>
 *f
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <!-- technical-bibtex-end -->
 *
 * @author Grigorios Tsoumakas
 */
public class HOMER extends MultiLabelMetaLearner {

    private final int numClusters;
    private HMC hmc;
    private HierarchyBuilder hb;
    private Instances header;
    private Method method;
    private MultiLabelInstances m;
    private int numMetaLabels;

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
    public long getNoNodes() {
        return hmc.getNoNodes();
    }

    public long getNoClassifierEvals() {
        return hmc.getNoClassifierEvals();
    }

    public long getTotalUsedTrainInsts() {
        return hmc.getTotalUsedTrainInsts();
    }
}
