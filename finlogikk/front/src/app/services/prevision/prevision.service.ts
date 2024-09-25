import { Injectable } from '@angular/core';
import { HttpClient } from "@angular/common/http";
import { Observable } from "rxjs";

@Injectable({
  providedIn: 'root'
})
export class PrevisionService {
  private baseUrl  = 'http://localhost:5000';

  constructor(private http: HttpClient) { }

  // getPrevisions(): Observable<any> {
  //   return this.http.get(`${this.baseUrl}/prevision`);
  // }
  // downloadFile(filename: string): string {
  //   return `${this.baseUrl}/download/${filename}`;
  // }
  // openFile(filename: string): string {
  //   return `${this.baseUrl}/openfile/${filename}`;
  // }

  // Récupérer la liste des fichiers PDF depuis la base de données
  getPrevisions(): Observable<any> {
    return this.http.get(`${this.baseUrl}/prevision`);
  }

  // Lien pour télécharger un fichier PDF à partir de la base de données
  downloadFile(filename: string): string {
    return `${this.baseUrl}/downloadfile/${filename}`;
  }

  // Lien pour ouvrir un fichier PDF dans le navigateur depuis la base de données
  openFile(filename: string): string {
    return `${this.baseUrl}/openfile/${filename}`;
  }

}
