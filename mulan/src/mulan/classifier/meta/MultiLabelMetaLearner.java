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
 *    MultiLabelMetaLearner.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.classifier.meta;

import mulan.classifier.transformation.*;
import mulan.classifier.*;
import weka.classifiers.trees.J48;

/**
 * Base class for multi-label learners, which use other multi-label learners
 *
 * @author Grigorios Tsoumakas
 * @version $Revision: 0.01 $
 */
public abstract class MultiLabelMetaLearner extends MultiLabelLearnerBase {

    /**
     * The encapsulated classifier or used for making clones in the
     * case of ensemble classifiers.
     */
    protected final MultiLabelLearner baseLearner;

    /**
     * Creates a new instance of {@link MultiLabelMetaLearner} with default
     * {@link LabelPowerset} multi-label classifier using J48 as the base
     * classifier.
     * @throws Exception 
     */
    public MultiLabelMetaLearner() throws Exception {
        this(new LabelPowerset(new J48()));
    }

    /**
     * Creates a new instance.
     *
     * @param baseLearner the base multi-label learner which will be used
     * internally to handle the data.
     */
    public MultiLabelMetaLearner(MultiLabelLearner baseLearner) {
        this.baseLearner = baseLearner;
    }

    /**
     * Returns the {@link MultiLabelLearner} which is used internally by the learner.
     *
     * @return the baseLearner
     */
    public MultiLabelLearner getBaseLearner() {
        return baseLearner;
    }
}
