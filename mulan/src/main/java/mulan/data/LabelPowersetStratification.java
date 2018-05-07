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
 *    LabelPowersetStratification.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.data;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.transformations.LabelPowersetTransformation;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;

/**
 * Class for stratifying data based on label combinations
 * 
 * @author Grigorios Tsoumakas
 * @version 2012.05.08
 */
public class LabelPowersetStratification implements Stratification, TechnicalInformationHandler {

    private int seed;

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(TechnicalInformation.Type.CONFERENCE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Sechidis, Konstantinos and Tsoumakas, Grigorios and Vlahavas, Ioannis");
        result.setValue(TechnicalInformation.Field.TITLE, "On the stratification of multi-label data");
        result.setValue(TechnicalInformation.Field.BOOKTITLE, "Proceedings of the 2011 European conference on Machine learning and knowledge discovery in databases - Volume Part III");
        result.setValue(TechnicalInformation.Field.SERIES, "ECML PKDD'11");
        result.setValue(TechnicalInformation.Field.YEAR, "2011");
        result.setValue(TechnicalInformation.Field.ISBN, "978-3-642-23807-9");
        result.setValue(TechnicalInformation.Field.LOCATION, "Athens, Greece");
        result.setValue(TechnicalInformation.Field.PAGES, "145--158");
        result.setValue(TechnicalInformation.Field.PUBLISHER, "Springer-Verlag");
        result.setValue(TechnicalInformation.Field.ADDRESS, "Berlin, Heidelberg");

        return result;
    }
    
    /**
     * Default constructor
     */
    public LabelPowersetStratification() {
        seed = 0;
    }
    
    /**
     * Constructor setting the random seed
     * 
     * @param aSeed the seed for random generation
     */    
    public LabelPowersetStratification(int aSeed) {
        seed = aSeed;
    }

    public MultiLabelInstances[] stratify(MultiLabelInstances data, int folds) {
        try {
            MultiLabelInstances[] segments = new MultiLabelInstances[folds];
            LabelPowersetTransformation transformation = new LabelPowersetTransformation();
            Instances transformed;

            // transform to single-label
            transformed = transformation.transformInstances(data);
            
            // add id 
            Add add = new Add();
            add.setAttributeIndex("first");
            add.setAttributeName("instanceID");
            add.setInputFormat(transformed);
            transformed = Filter.useFilter(transformed, add);
            for (int i=0; i<transformed.numInstances(); i++) {
                transformed.instance(i).setValue(0, i);
            }            
            transformed.setClassIndex(transformed.numAttributes()-1);
            
            // stratify
            transformed.randomize(new Random(seed));
            transformed.stratify(folds);
            
            for (int i = 0; i < folds; i++) {
                //System.out.println("Fold " + (i + 1) + "/" + folds);
                Instances temp = transformed.testCV(folds, i);
                Instances test = new Instances(data.getDataSet(), 0);
                for (int j=0; j<temp.numInstances(); j++) {
                    test.add(data.getDataSet().instance((int) temp.instance(j).value(0)));
                }                
                segments[i] = new MultiLabelInstances(test, data.getLabelsMetaData());
            }
            return segments;
        } catch (Exception ex) {
            Logger.getLogger(LabelPowersetStratification.class.getName()).log(Level.SEVERE, null, ex);
            return null;
        }
    }
}
