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
*    LabelsBuilder.java
*    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
*
*/

package mulan.core.data;

import java.io.File;

import javax.xml.XMLConstants;
import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Unmarshaller;
import javax.xml.bind.helpers.DefaultValidationEventHandler;
import javax.xml.validation.Schema;
import javax.xml.validation.SchemaFactory;

import org.xml.sax.SAXException;

/**
 * The {@link LabelsBuilder} is responsible for creation of {@link LabelsMetaDataImpl} instance 
 * from specified XML file source. The builder ensures XML source validity against XML schema. 
 * 
 * @author Jozef Vilcek
 */
public final class LabelsBuilder {

	private static final String LABELS_SCHEMA_SOURCE = "mulan/core/data/labels.xsd";
	protected static final String LABELS_SCHEMA_NAMESPACE = "http://mulan.sourceforge.net/labels";
	//private static final String LABELS_SCHEMA_LOCATION_ID = "http://mulan.sourceforge.net/schemata/labels.xsd";
	private static final String SCHEMA_FULL_CHECKING_FEATURE = "http://apache.org/xml/features/validation/schema-full-checking";

	
	/**
	 * Creates a {@link LabelsMetaDataImpl} instance form XML file specified by path
	 * 
	 * @param xmlLabelsFilePath the path to XML file containing labels definition
	 * @return a labels meta data instance 
	 * @throws IllegalArgumentException if input parameter is null
	 * @throws IllegalArgumentException if XML file does not exists under specified path
	 * @throws LabelsBuilderException if any error occur when validating XML against 
	 * 								  schema or when creating labels data
	 */
	public static LabelsMetaDataImpl createLabels(String xmlLabelsFilePath) throws LabelsBuilderException{
		
		if(xmlLabelsFilePath == null){
			throw new IllegalArgumentException("The xmlLabelsFilePath parameter is null.");
		}
		File xmlDefFile = new File(xmlLabelsFilePath);
		if(!xmlDefFile.exists()){
			throw new IllegalArgumentException(String.format(
					"The specified XML file source '%s' does not exist.", 
							xmlDefFile.getAbsolutePath()));
		}
		
		LabelsMetaDataImpl result = null;
		
		try {
			
			SchemaFactory schemaFactory = SchemaFactory.newInstance(XMLConstants.W3C_XML_SCHEMA_NS_URI);
			schemaFactory.setFeature(SCHEMA_FULL_CHECKING_FEATURE, false);
			Schema schema = schemaFactory.newSchema(ClassLoader.getSystemResource(LABELS_SCHEMA_SOURCE));
			
			JAXBContext context = JAXBContext.newInstance(LabelsMetaDataImpl.class,LabelNodeImpl.class);
			Unmarshaller unmarshaller = context.createUnmarshaller();
			unmarshaller.setEventHandler(new DefaultValidationEventHandler());
			unmarshaller.setSchema(schema);
			unmarshaller.setListener(new UnmarshallingProcessor());
   
			result = (LabelsMetaDataImpl) unmarshaller.unmarshal(xmlDefFile);
			   
		} catch (JAXBException exception) {
			throw new LabelsBuilderException("Error when trying to create objects structure from XML source.",
					exception);

		} catch (SAXException exception) {
			throw new LabelsBuilderException(
					"Error when creating schema instance to validate the XML source for labels creation.", 
							exception);
		}
		
		return result;
	}
	
	private static class UnmarshallingProcessor extends Unmarshaller.Listener {
	
		@Override
		public void afterUnmarshal(Object target, Object parent) {
			
			if(parent instanceof LabelNodeImpl && target instanceof LabelNodeImpl){
				((LabelNodeImpl)target).setParent((LabelNodeImpl)parent);
			}
			
			if(target instanceof LabelsMetaDataImpl && parent == null ){
				((LabelsMetaDataImpl)target).doReInit();
			}
        }
	}
}
