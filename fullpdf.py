import os
import PyPDF2

def lister_pdfs(dossier):
    # Liste tous les fichiers dans le dossier et filtre les PDF
    fichiers_pdfs = [fichier for fichier in os.listdir(dossier) if fichier.endswith('.pdf')]
    fichiers_pdfs.sort()  # Tri par ordre alphabétique
    return fichiers_pdfs

def fusionner_pdfs(pdf_liste, dossier, pdf_sortie):
    # Crée un objet PdfFileMerger
    merger = PyPDF2.PdfMerger()

    # Boucle à travers la liste des PDF à fusionner
    for pdf in pdf_liste:
        chemin_pdf = os.path.join(dossier, pdf)
        # Ouvre chaque fichier PDF en mode lecture binaire
        with open(chemin_pdf, 'rb') as fichier_pdf:
            # Ajoute le PDF au merger
            merger.append(fichier_pdf)

    # Écrit le fichier fusionné dans le fichier de sortie
    with open(pdf_sortie, 'wb') as fichier_sortie:
        merger.write(fichier_sortie)

    print(f"Les fichiers PDF ont été fusionnés en {pdf_sortie}")

# Exemple d'utilisation
dossier_pdfs = 'pdf'  # Remplace par le chemin réel de ton dossier
liste_pdfs = lister_pdfs(dossier_pdfs)
pdf_fusionne = 'resultat_fusion.pdf'

fusionner_pdfs(liste_pdfs, dossier_pdfs, pdf_fusionne)
