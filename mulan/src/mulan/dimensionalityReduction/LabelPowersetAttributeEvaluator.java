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
 *    LabelPowersetAttributeEvaluator.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.dimensionalityReduction;

import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.data.MultiLabelInstances;
import mulan.transformations.LabelPowersetTransformation;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Instances;

/**
 * Performs attribute evaluation using the label powerset transformation. For
 * more information, see <br/>
 * <br/>
 * K. Trohidis, G. Tsoumakas, G. Kalliris, I. Vlahavas. "Multilabel
 * Classification of Music into Emotions". Proc. 2008 International Conference
 * on Music Information Retrieval (ISMIR 2008)
 * </p>
 *
 * BibTeX:
 *
 * <pre>
 * &#064;inproceedings{trohidis+etal:2008,
 *      author =    {Trohidis, K. and Tsoumakas, G. and Kalliris, G. and Vlahavas, I.},
 *      title =     {Multilabel Classification of Music into Emotions},
 *      booktitle = {Proc. 9th International Conference on Music Information Retrieval (ISMIR 2008), Philadelphia, PA, USA, 2008},
 *      year =      {2008},
 *      location =  {Philadephia, PA, USA},
 * }
 * </pre>
 *
 * @author Grigorios Tsoumakas
 */
public class LabelPowersetAttributeEvaluator extends ASEvaluation implements AttributeEvaluator {

    /** The single-label attribute evaluator to use underneath */
    private ASEvaluation baseAttributeEvaluator;

    /** Constructor that uses an evaluator on a multi-label dataset 
     * @param x 
     * @param mlData */
    public LabelPowersetAttributeEvaluator(ASEvaluation x, MultiLabelInstances mlData) {
        baseAttributeEvaluator = x;
        LabelPowersetTransformation lpt = new LabelPowersetTransformation();
        Instances data;
        try {
            data = lpt.transformInstances(mlData);
            ((ASEvaluation) baseAttributeEvaluator).buildEvaluator(data);
        } catch (Exception ex) {
            Logger.getLogger(LabelPowersetAttributeEvaluator.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public double evaluateAttribute(int attribute) throws Exception {
        return ((AttributeEvaluator) baseAttributeEvaluator).evaluateAttribute(attribute);
    }

    @Override
    public void buildEvaluator(Instances arg0) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}
