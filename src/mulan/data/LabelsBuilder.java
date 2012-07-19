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
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.data;

import java.io.*;
import javax.xml.XMLConstants;
import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.bind.Unmarshaller;
import javax.xml.bind.helpers.DefaultValidationEventHandler;
import javax.xml.validation.Schema;
import javax.xml.validation.SchemaFactory;
import mulan.core.ArgumentNullException;
import org.xml.sax.SAXException;

/**
 * The {@link LabelsBuilder} is responsible for creation of {@link LabelsMetaDataImpl} instance 
 * from specified XML file source. The builder ensures XML source validity against XML schema. 
 * 
 * @author Jozef Vilcek
 * @version 2012.02.26
 */
public final class LabelsBuilder {

    private static final String LABELS_SCHEMA_SOURCE = "mulan/data/labels.xsd";
    /** The namespace of the schema for label representation */
    protected static final String LABELS_SCHEMA_NAMESPACE = "http://mulan.sourceforge.net/labels";
    //private static final String LABELS_SCHEMA_LOCATION_ID = "http://mulan.sourceforge.net/schemata/labels.xsd";
    private static final String SCHEMA_FULL_CHECKING_FEATURE = "http://apache.org/xml/features/validation/schema-full-checking";

    /**
     * Creates a {@link LabelsMetaData} instance from XML file specified by the path.
     *
     * @param xmlLabelsFilePath the path to XML file containing labels definition
     * @return the {@link LabelsMetaData} instance
     * @throws ArgumentNullException if specified path to XML file is null
     * @throws IllegalArgumentException if file under specified path does not exists
     * @throws LabelsBuilderException if specified file can not be read/opened
     */
    public static LabelsMetaData createLabels(String xmlLabelsFilePath) throws LabelsBuilderException {

        if (xmlLabelsFilePath == null) {
            throw new ArgumentNullException("xmlLabelsFilePath");
        }
        File xmlDefFile = new File(xmlLabelsFilePath);
        if (!xmlDefFile.exists()) {
            throw new IllegalArgumentException(String.format(
                    "The specified XML file source '%s' does not exist.",
                    xmlDefFile.getAbsolutePath()));
        }

        LabelsMetaData result;
        BufferedInputStream xmlFileInputStream = null;
        try {
            xmlFileInputStream = new BufferedInputStream(new FileInputStream(xmlDefFile));
            result = createLabels(xmlFileInputStream);
        } catch (FileNotFoundException e) {
            throw new LabelsBuilderException(
                    String.format("Error when creating input stream for the file under path: '%s'.",
                    xmlLabelsFilePath));
        } finally {
            if (xmlFileInputStream != null) {
                try {
                    xmlFileInputStream.close();
                } catch (IOException e) {
                }
            }
        }

        return result;
    }

    /**
     * Creates a {@link LabelsMetaData} instance from specified input stream.
     *
     * @param inputStream the input stream containing labels definition in XML format
     * @return the {@link LabelsMetaData} instance
     * @throws ArgumentNullException if specified input stream is null
     * @throws LabelsBuilderException if any error occur when validating XML against
     * 	schema or when creating labels data form specified input stream
     */
    public static LabelsMetaData createLabels(InputStream inputStream) throws LabelsBuilderException {

        if (inputStream == null) {
            throw new ArgumentNullException("inputStream");
        }

        LabelsMetaDataImpl result = null;

        try {

            SchemaFactory schemaFactory = SchemaFactory.newInstance(XMLConstants.W3C_XML_SCHEMA_NS_URI);
            schemaFactory.setFeature(SCHEMA_FULL_CHECKING_FEATURE, false);
            Schema schema = schemaFactory.newSchema(LabelsBuilder.class.getClassLoader().getResource(LABELS_SCHEMA_SOURCE));

            JAXBContext context = JAXBContext.newInstance(LabelsMetaDataImpl.class, LabelNodeImpl.class);
            Unmarshaller unmarshaller = context.createUnmarshaller();
            unmarshaller.setEventHandler(new DefaultValidationEventHandler());
            unmarshaller.setSchema(schema);
            unmarshaller.setListener(new UnmarshallingProcessor());

            result = (LabelsMetaDataImpl) unmarshaller.unmarshal(inputStream);

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

    /**
     * Dumps specified labels meta-data into the file in XML format.
     * If the file already exists, the content will be overwritten.
     *
     * @param labelsMetaData the meta-data which has to be dumped into the file
     * @param xmlDumpFilePath the path to the file where meta-data should be dumped
     * @throws LabelsBuilderException if specified file can not be read/opened
     * @throws ArgumentNullException if path to the file is not specified
     */
    public static void dumpLabels(LabelsMetaData labelsMetaData, String xmlDumpFilePath) throws LabelsBuilderException {

        if (xmlDumpFilePath == null) {
            throw new ArgumentNullException("xmlDumpFilePath");
        }
        File xmlDumpFile = new File(xmlDumpFilePath);
        boolean fileExists = xmlDumpFile.exists();

        BufferedOutputStream fileOutStream = null;
        try {
            if (!fileExists) {
                xmlDumpFile.createNewFile();
            }
            fileOutStream = new BufferedOutputStream(new FileOutputStream(xmlDumpFile));
            dumpLabels(labelsMetaData, fileOutStream);
        } catch (IOException exception) {
            if (!fileExists) {
                xmlDumpFile.delete();
            }
            throw new LabelsBuilderException("Error creating file output stream, to which labels meta-data has to be dumped.");
        } finally {
            if (fileOutStream != null) {
                try {
                    fileOutStream.close();
                } catch (IOException ex) {
                }
            }
        }
    }

    /**
     * Dumps specified labels meta-data, in XML format, into the specified {@link OutputStream}.
     *
     * @param labelsMetaData the meta-data which has to be dumped
     * @param outputStream the output stream where XML dup will be written
     * @throws LabelsBuilderException if error occurs during the serialization of meta-data to
     * 	the XML format of resulting XML is not valid against the schema
     * @throws ArgumentNullException if specified output strema is null.
     */
    public static void dumpLabels(LabelsMetaData labelsMetaData, OutputStream outputStream) throws LabelsBuilderException {

        if (outputStream == null) {
            throw new ArgumentNullException("outputStream");
        }
        if (!(labelsMetaData instanceof LabelsMetaDataImpl)) {
            throw new IllegalArgumentException(String.format("The specified implementation " +
                    "of labels meta data '%s' is not supported.", labelsMetaData.getClass().getName()));
        }

        try {
            SchemaFactory schemaFactory = SchemaFactory.newInstance(XMLConstants.W3C_XML_SCHEMA_NS_URI);
            schemaFactory.setFeature(SCHEMA_FULL_CHECKING_FEATURE, false);
            Schema schema = schemaFactory.newSchema(ClassLoader.getSystemResource(LABELS_SCHEMA_SOURCE));

            JAXBContext context = JAXBContext.newInstance(LabelsMetaDataImpl.class, LabelNodeImpl.class);
            Marshaller marshaller = context.createMarshaller();
            marshaller.setEventHandler(new DefaultValidationEventHandler());
            marshaller.setSchema(schema);

            marshaller.marshal(labelsMetaData, outputStream);

        } catch (JAXBException exception) {
            throw new LabelsBuilderException("Error when trying to dump labels meta-data objects structure to XML file.",
                    exception);

        } catch (SAXException exception) {
            throw new LabelsBuilderException(
                    "Error when creating schema instance to validate XML dump of labels meta-data objects structure.",
                    exception);
        }
    }

    private static class UnmarshallingProcessor extends Unmarshaller.Listener {

        @Override
        public void afterUnmarshal(Object target, Object parent) {

            if (parent instanceof LabelNodeImpl && target instanceof LabelNodeImpl) {
                ((LabelNodeImpl) target).setParent((LabelNodeImpl) parent);
            }

            if (target instanceof LabelsMetaDataImpl && parent == null) {
                ((LabelsMetaDataImpl) target).doReInit();
            }
        }
    }
}