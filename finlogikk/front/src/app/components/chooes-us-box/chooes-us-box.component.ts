import { Component, Input, OnInit } from '@angular/core';

@Component({
  selector: 'app-chooes-us-box',
  templateUrl: './chooes-us-box.component.html',
  styleUrls: ['./chooes-us-box.component.css'],
})
export class ChooesUsBoxComponent implements OnInit {
  @Input() data: any;

  ngOnInit(): void {
  }
}
