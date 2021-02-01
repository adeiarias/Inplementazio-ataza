package WEKA;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Random;

public class App {

    private static DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
    private static LocalDateTime now;
    private static ConverterUtils.DataSource source;
    private static Instances data;
    private static NaiveBayes naiveBayes = new NaiveBayes();

    public static void main(String[] args) throws Exception {
        if(args.length == 2){//Ez dira behar den parametro kopurua eman
            String erroreMezua = "\njava -jar estimateNaiveBayes5fCV.jar\n" +
                    "Helburua: emandako datuekin Naive Bayes-en kalitatearen estimazioa lortu 5-fCV eskemaren bidez eta datuei buruzko informazioa eman\n" +
                    "Argumentuak:\n" +
                    "1. Datu sortaren kokapena (path) .arff  formatuan (input). Aurre-baldintza: klasea azken atributuan egongo da.\n" +
                    "2. Emaitzak idazteko irteerako fitxategiaren path-a (output).";
            System.out.println(erroreMezua);
        }else{
            String path = "/home/adeiarias/Descargas/heart-c.arff";//arff fitxategiaren path-a
            String emaitza = "/home/adeiarias/Escritorio/output.txt";//output fitxategiaren path-a
            arffFitxategiaKargatu(path);
            datuSortaInformazioa();
            fitxategiaBete(emaitza,path);
            hold_out();
        }
    }

    private static void hold_out() throws Exception {
        //hold-out metodoa prestatu
        //1.PAUSUA -> RANDOMIZE
        Randomize randomize = new Randomize();
        randomize.setInputFormat(data);
        Instances randomInstances = Filter.useFilter(data,randomize);
        //2.PAUSUA -> SPLIT
        /*RemovePercentage removePercentage = new RemovePercentage();
        removePercentage.setInputFormat(randomInstances);
        removePercentage.setInvertSelection(true);
        removePercentage.setPercentage(70);
        Instances train = Filter.useFilter(randomInstances,removePercentage);

        Instances test = Filter.useFilter(randomInstances,removePercentage);*/
        int trainSize = (int) Math.round(randomInstances.numInstances() * 70 / 100);
        int testSize = randomInstances.numInstances() - trainSize;
        Instances train = new Instances(randomInstances, 0, trainSize);
        Instances test = new Instances(randomInstances, trainSize, testSize);
        //3.PAUSUA
        NaiveBayes naive = new NaiveBayes();
        train.setClassIndex(train.numAttributes()-1);
        naive.buildClassifier(train);
        System.out.println("train -> " + train.numInstances());
        System.out.println("test -> " + test.numInstances());

        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(naive,test);
        System.out.println("ACCURACY -> " + eval.pctCorrect());
        System.out.println("NOT ACCURACY -> " + eval.pctIncorrect());
        System.out.println("85.7143");
    }

    private static void fitxategiaBete(String emaitza, String path) throws Exception {
        //NaiveBayes entrenatu
        NaiveBayes naiveBayes = new NaiveBayes();
        data.setClassIndex(data.numAttributes()-1);
        naiveBayes.buildClassifier(data);

        //5-fold cross validation kalitatea estimatu
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(naiveBayes,data,5,new Random(1));

        //fitxategian datuak sartu
        now = LocalDateTime.now();
        FileWriter fitxategia = new FileWriter(emaitza);
        PrintWriter printWriter = new PrintWriter(fitxategia);
        printWriter.println("EMAITZAK:");
        printWriter.println("Exekuzio data -> " + dtf.format(now));
        printWriter.println("Fitxategiaren path-a -> " + path);
        printWriter.println("Nahasmen matrizea -> \n" + eval.toMatrixString());
        fitxategia.close();
    }

    private static void datuSortaInformazioa() {
        System.out.println("\nDATU SORTARI BURUZKO INFORMAZIOA:");
        System.out.println("Instantzia kopurua -> " + data.numInstances());
        System.out.println("Atributu kopurua -> " + data.numAttributes());
        System.out.println("Lehenengo atributuak har ditzakeen balio ezberdinak -> " + data.numDistinctValues(0));
        System.out.println("Azken aurreko atributuak dituen missing balio kopurua -> " + data.attributeStats(data.numAttributes()-2).missingCount + "\n");
    }

    private static void arffFitxategiaKargatu(String path) throws Exception {
        source = new ConverterUtils.DataSource(path);
        data = source.getDataSet();
        if (data.classIndex() == -1) data.setClassIndex(data.numAttributes() - 1);
    }
}
