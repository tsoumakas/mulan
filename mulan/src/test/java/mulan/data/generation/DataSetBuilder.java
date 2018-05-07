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
package mulan.data.generation;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;

import mulan.core.MulanRuntimeException;
import mulan.core.Util;
import mulan.data.InvalidDataFormatException;
import mulan.data.LabelNodeImpl;
import mulan.data.LabelsMetaDataImpl;
import mulan.data.MultiLabelInstances;
import mulan.data.generation.Attribute.AttributeType;
import weka.core.Instances;


/**
 * Helper class for building data sets used for unit testing purposes.
 * 
 * @author Jozef Vilcek
 */
public class DataSetBuilder {

	private static final String ARFF_MISSING_VALUE = "?";
	private static final String EMPTY_SPACE = " ";
  	private static final String VALUE_SEPARATOR = ",";
  	private static final String NEW_LINE = Util.getNewLineSeparator();

	
	public static MultiLabelInstances CreateDataSet(DataSetDefinition dataSetDefinition) throws InvalidDataFormatException{
		List<Attribute> attributes = dataSetDefinition.getAttributes();
		LabelsMetaDataImpl meta = new LabelsMetaDataImpl();
		for(Attribute attribute : attributes){
			if(attribute.isLabelAttribute()){
				meta.addRootNode(new LabelNodeImpl(attribute.getName()));
			}
		}
		
		InputStream arffStream = CreateArffDataSet(dataSetDefinition);
		Instances dataSet = null;
		try {
			dataSet = new Instances(new InputStreamReader(arffStream));
		} catch (IOException e) {
			throw new MulanRuntimeException("Unexpected error while creating data set.");
		}
		
		return new MultiLabelInstances(dataSet, meta);
	}
	
	public static InputStream CreateArffDataSet(DataSetDefinition dataSetDefinition){
	  	StringBuilder sw = new StringBuilder();
	  	sw.append(Instances.ARFF_RELATION).append(EMPTY_SPACE).append(dataSetDefinition.getName()).append(NEW_LINE);
	  	
	  	List<Attribute> attributes = dataSetDefinition.getAttributes();
	  	for(Attribute attr : attributes){
	  		sw.append(weka.core.Attribute.ARFF_ATTRIBUTE).append(EMPTY_SPACE);
	  		sw.append(attr.getName()).append(EMPTY_SPACE).append(getAttributeType(attr)).append(NEW_LINE);
	  	}
		
	  	sw.append(Instances.ARFF_DATA).append(NEW_LINE);
	  	
	  	// generate vectors
	  	for(int i = 0; i < dataSetDefinition.getExamplesCount(); i++){
	  		for(Attribute attribute : attributes){
	  			sw.append(getAttributeValue(attribute)).append(VALUE_SEPARATOR);
	  		}
	  		sw.deleteCharAt(sw.length() - 1);
	  		sw.append(NEW_LINE);
	  	}
	  	
		return new ByteArrayInputStream(sw.toString().getBytes());
	}
	
	
	private static String getAttributeType(Attribute attribute) {
		switch (attribute.getType()) {
		case Nominal:
			String[] nominalValues = ((NominalAttribute)attribute).getValues();
			StringBuilder sb = new StringBuilder();
			sb.append("{");
			for(String value : nominalValues){
				sb.append(value).append(VALUE_SEPARATOR);
			}
			sb.deleteCharAt(sb.length() - 1);
			sb.append("}");
			return sb.toString();
		case Numeric:
			return weka.core.Attribute.ARFF_ATTRIBUTE_NUMERIC;
		case String:
			return weka.core.Attribute.ARFF_ATTRIBUTE_STRING;
		default:
			throw new IllegalArgumentException(String.format("The specified attribute type '%s' " +
					"is not supported.", attribute.getType()));
		}
	}
	
	private static String getAttributeValue(Attribute attribute){
		double missingProb = attribute.getMissingValuesProbability();
		if(Math.random() < missingProb){
			return ARFF_MISSING_VALUE;
		}
		
		AttributeType type = attribute.getType();
		switch(type){
		case Nominal:
			return getAttributeValue((NominalAttribute)attribute);
		case Numeric:
			return getAttributeValue((NumericAttribute)attribute);
		case String:
			return getStringAttributeValue(attribute);
		default:
			throw new IllegalArgumentException(String.format("The specified attribute type '%s' " +
					"is not supported.", attribute.getType()));
		}
	}
	
	private static String getAttributeValue(NominalAttribute attribute){
		// random select a nominal value
		double rndValue = Math.random();
		double step = 1.0 / attribute.getValues().length;
		
		int index = 0;
		while(rndValue > step*(index + 1)){
			index++;
		}
		
		return attribute.getValues()[index];
	}
	
	private static String getAttributeValue(NumericAttribute attribute){
		// random select value from attribute provided range
		double value = Math.random() * (attribute.getMax() - attribute.getMin()) - attribute.getMin();
		return Double.toString(value);
	}
	
	private static String getStringAttributeValue(Attribute attribute){
		return "string_value";
	}
	
	


}


	
