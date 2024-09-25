import { Component, OnInit } from '@angular/core';
import { PrevisionService } from "../../services/prevision/prevision.service";
import {DomSanitizer, SafeResourceUrl} from "@angular/platform-browser";

@Component({
  selector: 'app-prevision',
  templateUrl: './prevision.component.html',
  styleUrls: ['./prevision.component.css']
})
export class PrevisionComponent {
  // files: { filename: string }[] = [];  // Updated to only store filenames
  // searchTerm: string = '';
  //
  // constructor(private previsionService: PrevisionService) { }
  //
  // ngOnInit(): void {
  //   this.previsionService.getPrevisions().subscribe(data => {
  //     this.files = data.files;
  //   });
  // }
  //
  // // Filtrer les fichiers en fonction du terme de recherche
  // filteredFiles(): { filename: string }[] {
  //   return this.files.filter(file => file.filename.toLowerCase().includes(this.searchTerm.toLowerCase()));
  // }
  //
  // // Construire le lien de téléchargement
  // downloadLink(file: { filename: string }): string {
  //   return this.previsionService.downloadFile(file.filename);
  // }
  //
  // // Construire le lien pour ouvrir le fichier
  // openLink(file: { filename: string }): string {
  //   return this.previsionService.openFile(file.filename);
  // }

  streamlitUrl: SafeResourceUrl;

  constructor(private sanitizer: DomSanitizer) {
    this.streamlitUrl = this.sanitizer.bypassSecurityTrustResourceUrl('http://localhost:8501');
  }

}
